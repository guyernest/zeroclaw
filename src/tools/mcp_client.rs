//! MCP client adapter — bridges external MCP server tools into ZeroClaw's `Tool` trait.
//!
//! Supports both **stdio** (child process) and **HTTP** transports via the PMCP SDK.

use crate::config::McpServerConfig;
use crate::config::McpTransportType;
use crate::tools::traits::{Tool, ToolResult};
use async_trait::async_trait;
use pmcp::shared::transport::TransportMessage;
use pmcp::shared::{StreamableHttpTransport, StreamableHttpTransportConfig, Transport};
use pmcp::{CallToolResult, ClientCapabilities, Content, StdioTransport, ToolInfo};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

// ── ChildProcessTransport ────────────────────────────────────────

/// A transport that communicates with an MCP server spawned as a child process.
///
/// Uses the same newline-delimited JSON framing as PMCP's `StdioTransport`,
/// but reads/writes the child's stdin/stdout instead of the current process's.
pub struct ChildProcessTransport {
    stdin: Mutex<BufWriter<ChildStdin>>,
    stdout: Mutex<BufReader<ChildStdout>>,
    child: Mutex<Child>,
    closed: AtomicBool,
}

impl fmt::Debug for ChildProcessTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChildProcessTransport")
            .field("closed", &self.closed.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ChildProcessTransport {
    /// Spawn an MCP server as a child process.
    pub fn spawn(config: &McpServerConfig) -> anyhow::Result<Self> {
        let command = config
            .command
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("MCP stdio server '{}' has no command", config.name))?;

        let mut cmd = tokio::process::Command::new(command);
        cmd.args(&config.args)
            .envs(&config.env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|e| {
            anyhow::anyhow!(
                "Failed to spawn MCP server '{}' (command: {}): {}",
                config.name,
                command,
                e
            )
        })?;

        let child_stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stdin of MCP server"))?;
        let child_stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stdout of MCP server"))?;

        // Drain stderr in background to prevent pipe buffer from blocking the child.
        if let Some(stderr) = child.stderr.take() {
            let name = config.name.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(mcp_server = %name, "[stderr] {}", line);
                }
            });
        }

        Ok(Self {
            stdin: Mutex::new(BufWriter::new(child_stdin)),
            stdout: Mutex::new(BufReader::new(child_stdout)),
            child: Mutex::new(child),
            closed: AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl Transport for ChildProcessTransport {
    async fn send(&mut self, message: TransportMessage) -> pmcp::Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(pmcp::Error::Transport(
                pmcp::error::TransportError::ConnectionClosed,
            ));
        }

        let json_bytes = StdioTransport::serialize_message(&message)?;
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(&json_bytes)
            .await
            .map_err(pmcp::Error::from)?;
        stdin.write_all(b"\n").await.map_err(pmcp::Error::from)?;
        stdin.flush().await.map_err(pmcp::Error::from)?;
        Ok(())
    }

    async fn receive(&mut self) -> pmcp::Result<TransportMessage> {
        if self.closed.load(Ordering::Acquire) {
            return Err(pmcp::Error::Transport(
                pmcp::error::TransportError::ConnectionClosed,
            ));
        }

        let mut stdout = self.stdout.lock().await;
        let mut line = String::new();
        let n = stdout
            .read_line(&mut line)
            .await
            .map_err(pmcp::Error::from)?;
        if n == 0 {
            self.closed.store(true, Ordering::Release);
            return Err(pmcp::Error::Transport(
                pmcp::error::TransportError::ConnectionClosed,
            ));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(pmcp::Error::Transport(
                pmcp::error::TransportError::InvalidMessage("Empty line received".into()),
            ));
        }
        StdioTransport::parse_message(trimmed.as_bytes())
    }

    async fn close(&mut self) -> pmcp::Result<()> {
        self.closed.store(true, Ordering::Release);
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        let _ = tokio::time::timeout(std::time::Duration::from_secs(3), child.wait()).await;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        !self.closed.load(Ordering::Acquire)
    }

    fn transport_type(&self) -> &'static str {
        "child-process"
    }
}

// ── McpTransport enum ────────────────────────────────────────────

/// Unified transport enum so `McpClientAdapter` holds a non-generic `Client<McpTransport>`.
#[derive(Debug)]
pub(crate) enum McpTransport {
    ChildProcess(ChildProcessTransport),
    Http(StreamableHttpTransport),
}

#[async_trait]
impl Transport for McpTransport {
    async fn send(&mut self, message: TransportMessage) -> pmcp::Result<()> {
        match self {
            Self::ChildProcess(t) => t.send(message).await,
            Self::Http(t) => t.send(message).await,
        }
    }

    async fn receive(&mut self) -> pmcp::Result<TransportMessage> {
        match self {
            Self::ChildProcess(t) => t.receive().await,
            Self::Http(t) => t.receive().await,
        }
    }

    async fn close(&mut self) -> pmcp::Result<()> {
        match self {
            Self::ChildProcess(t) => t.close().await,
            Self::Http(t) => t.close().await,
        }
    }

    fn is_connected(&self) -> bool {
        match self {
            Self::ChildProcess(t) => t.is_connected(),
            Self::Http(t) => t.is_connected(),
        }
    }

    fn transport_type(&self) -> &'static str {
        match self {
            Self::ChildProcess(t) => t.transport_type(),
            Self::Http(t) => t.transport_type(),
        }
    }
}

// ── McpClientAdapter ─────────────────────────────────────────────

/// Wraps a PMCP `Client` and provides lifecycle management (connect, call, reconnect).
pub struct McpClientAdapter {
    config: McpServerConfig,
    client: tokio::sync::RwLock<pmcp::Client<McpTransport>>,
    tools: Vec<ToolInfo>,
    restart_count: AtomicU32,
}

impl fmt::Debug for McpClientAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpClientAdapter")
            .field("name", &self.config.name)
            .field("transport", &self.config.transport)
            .field("tools", &self.tools.len())
            .field("restart_count", &self.restart_count.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl McpClientAdapter {
    /// Connect to an MCP server and discover its tools.
    pub async fn connect(config: &McpServerConfig) -> anyhow::Result<Arc<Self>> {
        let transport = Self::build_transport(config)?;
        let mut client = pmcp::Client::new(transport);

        let init_timeout = std::time::Duration::from_secs(config.init_timeout_secs);
        tokio::time::timeout(
            init_timeout,
            client.initialize(ClientCapabilities::default()),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "MCP server '{}' initialization timed out after {}s",
                config.name,
                config.init_timeout_secs
            )
        })?
        .map_err(|e| {
            anyhow::anyhow!("MCP server '{}' initialization failed: {}", config.name, e)
        })?;

        let list_result = client.list_tools(None).await.map_err(|e| {
            anyhow::anyhow!("MCP server '{}' failed to list tools: {}", config.name, e)
        })?;

        let tools = filter_tools(
            list_result.tools,
            &config.allowed_tools,
            &config.blocked_tools,
        );

        crate::health::mark_component_ok(&format!("mcp:{}", config.name));

        Ok(Arc::new(Self {
            config: config.clone(),
            client: tokio::sync::RwLock::new(client),
            tools,
            restart_count: AtomicU32::new(0),
        }))
    }

    fn build_transport(config: &McpServerConfig) -> anyhow::Result<McpTransport> {
        match config.transport {
            McpTransportType::Stdio => {
                let t = ChildProcessTransport::spawn(config)?;
                Ok(McpTransport::ChildProcess(t))
            }
            McpTransportType::Http => {
                let url_str = config.url.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("MCP HTTP server '{}' has no url", config.name)
                })?;
                let url = url::Url::parse(url_str).map_err(|e| {
                    anyhow::anyhow!(
                        "MCP server '{}' invalid url '{}': {}",
                        config.name,
                        url_str,
                        e
                    )
                })?;
                let http_config = StreamableHttpTransportConfig {
                    url,
                    extra_headers: config
                        .headers
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                    auth_provider: None,
                    session_id: None,
                    enable_json_response: true,
                    on_resumption_token: None,
                    http_middleware_chain: None,
                };
                let t = StreamableHttpTransport::new(http_config);
                Ok(McpTransport::Http(t))
            }
        }
    }

    /// Call a tool on the MCP server.
    async fn call_tool(&self, name: &str, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let timeout = std::time::Duration::from_secs(self.config.tool_call_timeout_secs);
        let client = self.client.read().await;
        let result = tokio::time::timeout(timeout, client.call_tool(name.to_string(), args))
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "MCP tool call '{}' on server '{}' timed out after {}s",
                    name,
                    self.config.name,
                    self.config.tool_call_timeout_secs,
                )
            })?
            .map_err(|e| {
                anyhow::anyhow!(
                    "MCP tool call '{}' on server '{}' failed: {}",
                    name,
                    self.config.name,
                    e
                )
            })?;

        Ok(map_call_tool_result(&result))
    }

    /// Attempt to reconnect a stdio transport after failure.
    async fn reconnect(&self) -> anyhow::Result<()> {
        if self.config.transport != McpTransportType::Stdio {
            return Ok(()); // HTTP is stateless, no reconnect needed
        }

        let count = self.restart_count.fetch_add(1, Ordering::SeqCst) + 1;
        if !self.config.restart_on_crash || count > self.config.max_restarts {
            return Err(anyhow::anyhow!(
                "MCP server '{}' exceeded max restarts ({})",
                self.config.name,
                self.config.max_restarts
            ));
        }

        crate::health::bump_component_restart(&format!("mcp:{}", self.config.name));
        tracing::warn!(
            "MCP server '{}': reconnecting (attempt {}/{})",
            self.config.name,
            count,
            self.config.max_restarts
        );

        let transport = Self::build_transport(&self.config)?;
        let mut new_client = pmcp::Client::new(transport);

        let init_timeout = std::time::Duration::from_secs(self.config.init_timeout_secs);
        tokio::time::timeout(
            init_timeout,
            new_client.initialize(ClientCapabilities::default()),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "MCP server '{}' reconnect initialization timed out",
                self.config.name
            )
        })?
        .map_err(|e| {
            anyhow::anyhow!(
                "MCP server '{}' reconnect initialization failed: {}",
                self.config.name,
                e
            )
        })?;

        let mut client = self.client.write().await;
        *client = new_client;

        crate::health::mark_component_ok(&format!("mcp:{}", self.config.name));
        Ok(())
    }

    /// Call a tool with automatic recovery on transport errors (stdio only).
    pub async fn call_tool_with_recovery(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        match self.call_tool(name, args.clone()).await {
            Ok(result) => Ok(result),
            Err(e) => {
                tracing::warn!(
                    "MCP tool call '{}' on server '{}' failed: {}; attempting recovery",
                    name,
                    self.config.name,
                    e
                );
                self.reconnect().await?;
                self.call_tool(name, args).await
            }
        }
    }

    /// Convert discovered tools into ZeroClaw `Tool` trait objects.
    pub fn into_tools(self: &Arc<Self>) -> Vec<Box<dyn Tool>> {
        self.tools
            .iter()
            .map(|info| {
                let proxy: Box<dyn Tool> = Box::new(McpToolProxy {
                    adapter: Arc::clone(self),
                    namespaced_name: format!("{}.{}", self.config.name, info.name),
                    original_name: info.name.clone(),
                    description: info
                        .description
                        .clone()
                        .unwrap_or_else(|| format!("MCP tool: {}", info.name)),
                    input_schema: info.input_schema.clone(),
                });
                proxy
            })
            .collect()
    }
}

// ── McpToolProxy ─────────────────────────────────────────────────

/// Thin proxy that implements ZeroClaw's `Tool` trait for an MCP tool.
pub struct McpToolProxy {
    adapter: Arc<McpClientAdapter>,
    namespaced_name: String,
    original_name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[async_trait]
impl Tool for McpToolProxy {
    fn name(&self) -> &str {
        &self.namespaced_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.input_schema.clone()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.adapter
            .call_tool_with_recovery(&self.original_name, args)
            .await
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Filter discovered tools by allowed/blocked lists.
fn filter_tools(tools: Vec<ToolInfo>, allowed: &[String], blocked: &[String]) -> Vec<ToolInfo> {
    tools
        .into_iter()
        .filter(|t| {
            if !allowed.is_empty() && !allowed.contains(&t.name) {
                return false;
            }
            if blocked.contains(&t.name) {
                return false;
            }
            true
        })
        .collect()
}

/// Map a PMCP `CallToolResult` to ZeroClaw's `ToolResult`.
fn map_call_tool_result(result: &CallToolResult) -> ToolResult {
    let mut parts: Vec<String> = Vec::new();

    for content in &result.content {
        match content {
            Content::Text { text } => parts.push(text.clone()),
            Content::Image { mime_type, .. } => {
                parts.push(format!("[image: {}]", mime_type));
            }
            Content::Resource { uri, .. } => {
                parts.push(format!("[resource: {}]", uri));
            }
        }
    }

    let output = parts.join("\n");

    if result.is_error {
        ToolResult {
            success: false,
            output: String::new(),
            error: Some(output),
        }
    } else {
        ToolResult {
            success: true,
            output,
            error: None,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{McpServerConfig, McpTransportType};
    use std::collections::HashMap;

    fn stdio_config() -> McpServerConfig {
        McpServerConfig {
            name: "test-stdio".into(),
            transport: McpTransportType::Stdio,
            command: Some("echo".into()),
            args: vec!["hello".into()],
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
            enabled: true,
            allowed_tools: Vec::new(),
            blocked_tools: Vec::new(),
            restart_on_crash: true,
            max_restarts: 3,
            init_timeout_secs: 30,
            tool_call_timeout_secs: 60,
        }
    }

    fn http_config() -> McpServerConfig {
        McpServerConfig {
            name: "test-http".into(),
            transport: McpTransportType::Http,
            command: None,
            args: Vec::new(),
            env: HashMap::new(),
            url: Some("http://localhost:3100/mcp".into()),
            headers: {
                let mut h = HashMap::new();
                h.insert("Authorization".into(), "Bearer tok".into());
                h
            },
            enabled: true,
            allowed_tools: vec!["navigate".into(), "click".into()],
            blocked_tools: Vec::new(),
            restart_on_crash: true,
            max_restarts: 3,
            init_timeout_secs: 30,
            tool_call_timeout_secs: 60,
        }
    }

    // ── Tool filtering tests ──────────────────────────────────

    fn make_tool_info(name: &str) -> ToolInfo {
        ToolInfo::new(
            name,
            Some(format!("Tool {name}")),
            serde_json::json!({"type": "object"}),
        )
    }

    #[test]
    fn filter_tools_allows_all_when_empty() {
        let tools = vec![
            make_tool_info("a"),
            make_tool_info("b"),
            make_tool_info("c"),
        ];
        let result = filter_tools(tools, &[], &[]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn filter_tools_allowed_whitelist() {
        let tools = vec![
            make_tool_info("a"),
            make_tool_info("b"),
            make_tool_info("c"),
        ];
        let result = filter_tools(tools, &["a".into(), "c".into()], &[]);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["a", "c"]);
    }

    #[test]
    fn filter_tools_blocked_blacklist() {
        let tools = vec![
            make_tool_info("a"),
            make_tool_info("b"),
            make_tool_info("c"),
        ];
        let result = filter_tools(tools, &[], &["b".into()]);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["a", "c"]);
    }

    #[test]
    fn filter_tools_both_allowed_and_blocked() {
        let tools = vec![
            make_tool_info("a"),
            make_tool_info("b"),
            make_tool_info("c"),
        ];
        let result = filter_tools(tools, &["a".into(), "b".into()], &["b".into()]);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["a"]);
    }

    // ── Namespacing tests ─────────────────────────────────────

    #[test]
    fn tool_namespacing_format() {
        let name = format!("{}.{}", "browser", "navigate");
        assert_eq!(name, "browser.navigate");
    }

    // ── ToolResult mapping tests ──────────────────────────────

    #[test]
    fn map_text_content() {
        let result = CallToolResult::new(vec![Content::Text {
            text: "hello world".into(),
        }]);
        let mapped = map_call_tool_result(&result);
        assert!(mapped.success);
        assert_eq!(mapped.output, "hello world");
        assert!(mapped.error.is_none());
    }

    #[test]
    fn map_error_response() {
        let result = CallToolResult::error(vec![Content::Text {
            text: "something went wrong".into(),
        }]);
        let mapped = map_call_tool_result(&result);
        assert!(!mapped.success);
        assert!(mapped.output.is_empty());
        assert_eq!(mapped.error.as_deref(), Some("something went wrong"));
    }

    #[test]
    fn map_mixed_content() {
        let result = CallToolResult::new(vec![
            Content::Text {
                text: "result:".into(),
            },
            Content::Image {
                data: "base64data".into(),
                mime_type: "image/png".into(),
            },
            Content::Resource {
                uri: "file:///tmp/out.txt".into(),
                text: None,
                mime_type: None,
            },
        ]);
        let mapped = map_call_tool_result(&result);
        assert!(mapped.success);
        assert_eq!(
            mapped.output,
            "result:\n[image: image/png]\n[resource: file:///tmp/out.txt]"
        );
    }

    #[test]
    fn map_empty_content() {
        let result = CallToolResult::new(vec![]);
        let mapped = map_call_tool_result(&result);
        assert!(mapped.success);
        assert!(mapped.output.is_empty());
    }

    // ── Config defaults / deserialization tests ───────────────

    #[test]
    fn config_defaults_mcp_servers_empty() {
        let config = crate::config::Config::default();
        assert!(config.mcp_servers.is_empty());
    }

    #[test]
    fn config_toml_backward_compat_no_mcp_servers() {
        // A minimal TOML with no mcp_servers should deserialize fine
        let toml_str = r#"
default_temperature = 0.7
"#;
        let config: crate::config::Config = toml::from_str(toml_str).unwrap();
        assert!(config.mcp_servers.is_empty());
    }

    #[test]
    fn config_toml_with_mcp_servers() {
        let toml_str = r#"
default_temperature = 0.7

[[mcp_servers]]
name = "browser"
transport = "http"
url = "http://localhost:3100/mcp"
allowed_tools = ["navigate", "click"]

[[mcp_servers]]
name = "desktop"
transport = "stdio"
command = "mcp-desktop-server"
args = ["--display", ":0"]
"#;
        let config: crate::config::Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.mcp_servers.len(), 2);

        let browser = &config.mcp_servers[0];
        assert_eq!(browser.name, "browser");
        assert_eq!(browser.transport, McpTransportType::Http);
        assert_eq!(browser.url.as_deref(), Some("http://localhost:3100/mcp"));
        assert_eq!(browser.allowed_tools, vec!["navigate", "click"]);
        assert!(browser.enabled); // default_true

        let desktop = &config.mcp_servers[1];
        assert_eq!(desktop.name, "desktop");
        assert_eq!(desktop.transport, McpTransportType::Stdio);
        assert_eq!(desktop.command.as_deref(), Some("mcp-desktop-server"));
        assert_eq!(desktop.args, vec!["--display", ":0"]);
        assert!(desktop.restart_on_crash); // default_true
        assert_eq!(desktop.max_restarts, 3);
        assert_eq!(desktop.init_timeout_secs, 30);
        assert_eq!(desktop.tool_call_timeout_secs, 60);
    }

    #[test]
    fn config_toml_defaults_applied() {
        let toml_str = r#"
default_temperature = 0.7

[[mcp_servers]]
name = "minimal"
"#;
        let config: crate::config::Config = toml::from_str(toml_str).unwrap();
        let server = &config.mcp_servers[0];
        assert_eq!(server.transport, McpTransportType::Http); // default
        assert!(server.enabled);
        assert!(server.restart_on_crash);
        assert_eq!(server.max_restarts, 3);
        assert_eq!(server.init_timeout_secs, 30);
        assert_eq!(server.tool_call_timeout_secs, 60);
        assert!(server.allowed_tools.is_empty());
        assert!(server.blocked_tools.is_empty());
    }

    // ── McpTransport enum delegation tests ────────────────────

    #[test]
    fn child_process_transport_debug() {
        // Just ensure Debug is implemented (compilation test)
        let config = stdio_config();
        // We can't actually spawn in tests without a real MCP server,
        // but we can test the config construction.
        assert_eq!(config.name, "test-stdio");
    }

    #[test]
    fn http_config_builds_url() {
        let config = http_config();
        let url = url::Url::parse(config.url.as_deref().unwrap()).unwrap();
        assert_eq!(url.host_str(), Some("localhost"));
        assert_eq!(url.port(), Some(3100));
        assert_eq!(url.path(), "/mcp");
    }

    // ── Live integration test ─────────────────────────────────
    // Run with: cargo test live_imdb_mcp -- --ignored --nocapture

    #[tokio::test]
    #[ignore] // requires network access
    async fn live_imdb_mcp_connect_and_call() {
        let config = McpServerConfig {
            name: "imdb".into(),
            transport: McpTransportType::Http,
            command: None,
            args: Vec::new(),
            env: HashMap::new(),
            url: Some("https://imdb.us-east.true-mcp.com/mcp".into()),
            headers: HashMap::new(),
            enabled: true,
            allowed_tools: Vec::new(),
            blocked_tools: Vec::new(),
            restart_on_crash: false,
            max_restarts: 0,
            init_timeout_secs: 30,
            tool_call_timeout_secs: 30,
        };

        // 1. Connect and discover tools
        let adapter = McpClientAdapter::connect(&config)
            .await
            .expect("Failed to connect to IMDB MCP server");

        println!("Connected to IMDB MCP server");
        println!("Discovered {} tool(s):", adapter.tools.len());
        for tool in &adapter.tools {
            println!(
                "  - {} : {}",
                tool.name,
                tool.description.as_deref().unwrap_or("(no description)")
            );
        }
        assert!(!adapter.tools.is_empty(), "Expected at least one tool");

        // 2. Convert to ZeroClaw Tool trait objects
        let tools = adapter.into_tools();
        println!("\nZeroClaw tool proxies:");
        for tool in &tools {
            println!("  - {}: {}", tool.name(), tool.description());
        }

        // 3. Try calling a tool (search for a well-known movie)
        let search_tool = tools.iter().find(|t| t.name().contains("search"));
        if let Some(tool) = search_tool {
            println!("\nCalling tool: {}", tool.name());
            let result = tool
                .execute(serde_json::json!({"query": "The Matrix"}))
                .await
                .expect("Tool call failed");
            println!("Success: {}", result.success);
            println!("Output: {}", &result.output[..result.output.len().min(500)]);
            assert!(result.success);
            assert!(!result.output.is_empty());
        } else {
            println!("\nNo search tool found, trying first available tool");
            let first = &tools[0];
            println!("Tool schema: {}", first.parameters_schema());
        }
    }
}
