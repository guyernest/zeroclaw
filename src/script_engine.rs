//! ScriptEngine — executes cloud-authored deterministic workflows via MCP tools.
//!
//! When the Activity channel (or any channel) receives a JSON payload containing
//! a `"script"` field, the ScriptEngine bypasses the local LLM agent loop and
//! dispatches each step directly to the appropriate MCP tool. This implements
//! the Split-Brain LLM Model (Section 11) from the architecture doc.
//!
//! Three step types are supported:
//! - **Tool steps** (`"tool": "server.tool_name"`) — direct MCP tool call
//! - **Code-mode steps** (`"tool_mode": "code"`) — calls `validate_code` then `execute_code`
//! - **Prompt steps** (`"prompt": "..."`) — proactive LLM invocation with optional context

use crate::config::ScriptEngineConfig;
use crate::observability::Observer;
use crate::providers::Provider;
use crate::tools::{Tool, ToolResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ── Data Structures ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Script {
    pub version: String,
    pub name: String,
    #[serde(default)]
    pub variables: serde_json::Value,
    pub steps: Vec<ScriptStep>,
    #[serde(default)]
    pub output: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptStep {
    pub id: String,
    // Tier 1: direct tool call
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub args: Option<serde_json::Value>,
    // Tier 2: code-mode
    #[serde(default)]
    pub tool_mode: Option<String>,
    #[serde(default)]
    pub server: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    // Tier 3: LLM prompt step
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub context_from: Option<Vec<String>>,
    // Common
    #[serde(default = "default_step_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub output_key: Option<String>,
    #[serde(default)]
    pub on_failure: Option<FailureStrategy>,
}

fn default_step_timeout() -> u64 {
    60
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureStrategy {
    Abort,
    Skip,
    Retry,
    LlmAssist,
}

// ── Result Structures ────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ScriptResult {
    pub success: bool,
    pub output: serde_json::Value,
    pub execution_details: ExecutionDetails,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ExecutionDetails {
    pub steps_executed: usize,
    pub steps_succeeded: usize,
    pub steps_failed: usize,
    pub llm_assists: Vec<LlmAssistRecord>,
    pub duration_ms: u64,
    pub step_timings: Vec<StepTiming>,
}

#[derive(Debug, Serialize)]
pub struct LlmAssistRecord {
    pub step_id: String,
    pub error: String,
    pub resolution: String,
    pub resolved: bool,
}

#[derive(Debug, Serialize)]
pub struct StepTiming {
    pub step_id: String,
    pub duration_ms: u64,
    pub success: bool,
}

// ── ScriptEngine ─────────────────────────────────────────────────

pub struct ScriptEngine;

impl ScriptEngine {
    /// Try to parse a channel message as a script. Returns `None` if not a script.
    pub fn detect(content: &str) -> Option<Script> {
        let value: serde_json::Value = serde_json::from_str(content).ok()?;
        if let Some(script_value) = value.get("script") {
            serde_json::from_value(script_value.clone()).ok()
        } else if value.get("version").is_some() && value.get("steps").is_some() {
            serde_json::from_value(value).ok()
        } else {
            None
        }
    }

    /// Execute a script by dispatching steps to MCP tools.
    pub async fn execute(
        script: &Script,
        tools_registry: &[Box<dyn Tool>],
        provider: Option<&dyn Provider>,
        model: &str,
        _observer: &dyn Observer,
        config: &ScriptEngineConfig,
    ) -> ScriptResult {
        let started = Instant::now();
        let mut step_results: HashMap<String, serde_json::Value> = HashMap::new();
        let mut step_timings = Vec::with_capacity(script.steps.len());
        let mut llm_assists = Vec::new();
        let mut steps_succeeded = 0_usize;
        let mut steps_failed = 0_usize;
        let mut last_error: Option<String> = None;

        // Wrap the entire script in a timeout
        let script_deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(config.script_timeout_secs);

        for step in &script.steps {
            let step_start = Instant::now();
            let step_timeout = std::time::Duration::from_secs(step.timeout_secs);

            let result = tokio::time::timeout_at(
                // Use the earlier of step timeout and script deadline
                script_deadline.min(tokio::time::Instant::now() + step_timeout),
                Self::execute_step(step, tools_registry, &step_results, script, provider, model),
            )
            .await;

            let step_duration_ms = elapsed_ms(&step_start);
            let step_outcome = match result {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(e)) => Err(e.to_string()),
                Err(_) => Err(format!("Step '{}' timed out after {}s", step.id, step.timeout_secs)),
            };

            match step_outcome {
                Ok(value) => {
                    step_timings.push(StepTiming {
                        step_id: step.id.clone(),
                        duration_ms: step_duration_ms,
                        success: true,
                    });
                    step_results.insert(step.id.clone(), value);
                    steps_succeeded += 1;
                }
                Err(error) => {
                    let strategy = step.on_failure.as_ref().unwrap_or(&FailureStrategy::Abort);
                    match strategy {
                        FailureStrategy::Abort => {
                            step_timings.push(StepTiming {
                                step_id: step.id.clone(),
                                duration_ms: step_duration_ms,
                                success: false,
                            });
                            steps_failed += 1;
                            last_error = Some(format!("Step '{}' failed: {}", step.id, error));
                            break;
                        }
                        FailureStrategy::Skip => {
                            tracing::warn!("Script step '{}' failed (skipping): {}", step.id, error);
                            step_timings.push(StepTiming {
                                step_id: step.id.clone(),
                                duration_ms: step_duration_ms,
                                success: false,
                            });
                            step_results.insert(step.id.clone(), serde_json::Value::Null);
                            steps_failed += 1;
                        }
                        FailureStrategy::Retry => {
                            let mut retried = false;
                            for attempt in 1..=config.max_step_retries {
                                tracing::info!(
                                    "Retrying step '{}' (attempt {}/{})",
                                    step.id,
                                    attempt,
                                    config.max_step_retries
                                );
                                let retry_result = tokio::time::timeout_at(
                                    script_deadline.min(tokio::time::Instant::now() + step_timeout),
                                    Self::execute_step(step, tools_registry, &step_results, script, provider, model),
                                )
                                .await;

                                match retry_result {
                                    Ok(Ok(value)) => {
                                        step_results.insert(step.id.clone(), value);
                                        retried = true;
                                        steps_succeeded += 1;
                                        break;
                                    }
                                    Ok(Err(e)) => {
                                        tracing::warn!(
                                            "Retry {attempt} for step '{}' failed: {e}",
                                            step.id
                                        );
                                    }
                                    Err(_) => {
                                        tracing::warn!(
                                            "Retry {attempt} for step '{}' timed out",
                                            step.id
                                        );
                                    }
                                }
                            }
                            let retry_duration =
                                elapsed_ms(&step_start);
                            step_timings.push(StepTiming {
                                step_id: step.id.clone(),
                                duration_ms: retry_duration,
                                success: retried,
                            });
                            if !retried {
                                steps_failed += 1;
                                last_error = Some(format!(
                                    "Step '{}' failed after {} retries: {}",
                                    step.id, config.max_step_retries, error
                                ));
                                break;
                            }
                        }
                        FailureStrategy::LlmAssist => {
                            if !config.llm_fallback {
                                step_timings.push(StepTiming {
                                    step_id: step.id.clone(),
                                    duration_ms: step_duration_ms,
                                    success: false,
                                });
                                steps_failed += 1;
                                last_error = Some(format!(
                                    "Step '{}' failed (llm_fallback disabled): {}",
                                    step.id, error
                                ));
                                break;
                            }

                            let assist_result = Self::llm_assist(
                                step,
                                &error,
                                &step_results,
                                tools_registry,
                                provider,
                                model,
                                script,
                                script_deadline,
                            )
                            .await;

                            let assist_duration =
                                elapsed_ms(&step_start);

                            match assist_result {
                                Ok((value, resolution)) => {
                                    llm_assists.push(LlmAssistRecord {
                                        step_id: step.id.clone(),
                                        error: error.clone(),
                                        resolution,
                                        resolved: true,
                                    });
                                    step_timings.push(StepTiming {
                                        step_id: step.id.clone(),
                                        duration_ms: assist_duration,
                                        success: true,
                                    });
                                    step_results.insert(step.id.clone(), value);
                                    steps_succeeded += 1;
                                }
                                Err(assist_err) => {
                                    llm_assists.push(LlmAssistRecord {
                                        step_id: step.id.clone(),
                                        error: error.clone(),
                                        resolution: assist_err.clone(),
                                        resolved: false,
                                    });
                                    step_timings.push(StepTiming {
                                        step_id: step.id.clone(),
                                        duration_ms: assist_duration,
                                        success: false,
                                    });
                                    steps_failed += 1;
                                    last_error = Some(format!(
                                        "Step '{}' failed after LLM assist: {}",
                                        step.id, assist_err
                                    ));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Resolve output template
        let output = if let Some(output_template) = &script.output {
            let mut resolved = serde_json::Map::new();
            for (key, tmpl) in output_template {
                resolved.insert(
                    key.clone(),
                    resolve_templates(tmpl, &script.variables, &step_results),
                );
            }
            serde_json::Value::Object(resolved)
        } else {
            // No output template — return all step results
            serde_json::to_value(&step_results).unwrap_or(serde_json::Value::Null)
        };

        let success = last_error.is_none();
        ScriptResult {
            success,
            output,
            execution_details: ExecutionDetails {
                steps_executed: steps_succeeded + steps_failed,
                steps_succeeded,
                steps_failed,
                llm_assists,
                duration_ms: elapsed_ms(&started),
                step_timings,
            },
            error: last_error,
        }
    }

    /// Execute a single step.
    async fn execute_step(
        step: &ScriptStep,
        tools_registry: &[Box<dyn Tool>],
        step_results: &HashMap<String, serde_json::Value>,
        script: &Script,
        provider: Option<&dyn Provider>,
        model: &str,
    ) -> anyhow::Result<serde_json::Value> {
        if let Some(ref tool_name) = step.tool {
            // Tier 1: direct tool call
            Self::execute_tool_step(tool_name, step, tools_registry, step_results, script).await
        } else if step.tool_mode.as_deref() == Some("code") {
            // Tier 2: code-mode
            Self::execute_code_step(step, tools_registry, step_results, script).await
        } else if step.prompt.is_some() {
            // Tier 3: LLM prompt
            Self::execute_prompt_step(step, step_results, script, provider, model).await
        } else {
            anyhow::bail!(
                "Step '{}' has no 'tool', 'tool_mode: code', or 'prompt'",
                step.id
            );
        }
    }

    /// Tier 1: Direct tool call via `find_tool()`.
    async fn execute_tool_step(
        tool_name: &str,
        step: &ScriptStep,
        tools_registry: &[Box<dyn Tool>],
        step_results: &HashMap<String, serde_json::Value>,
        script: &Script,
    ) -> anyhow::Result<serde_json::Value> {
        let tool = find_tool(tools_registry, tool_name)
            .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found in registry", tool_name))?;

        let args = step
            .args
            .clone()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let resolved_args = resolve_templates(&args, &script.variables, step_results);

        let result = tool.execute(resolved_args).await?;
        tool_result_to_value(&result, step)
    }

    /// Tier 2: Code-mode via `validate_code` + `execute_code` on target MCP server.
    async fn execute_code_step(
        step: &ScriptStep,
        tools_registry: &[Box<dyn Tool>],
        step_results: &HashMap<String, serde_json::Value>,
        script: &Script,
    ) -> anyhow::Result<serde_json::Value> {
        let server = step
            .server
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Code-mode step '{}' missing 'server' field", step.id))?;
        let code = step
            .code
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Code-mode step '{}' missing 'code' field", step.id))?;

        // Resolve templates in the code string
        let resolved_code_value =
            resolve_templates(&serde_json::Value::String(code.to_string()), &script.variables, step_results);
        let resolved_code = resolved_code_value.as_str().unwrap_or(code);

        // Step 1: validate_code
        let validate_tool_name = format!("{server}.validate_code");
        let validate_tool = find_tool(tools_registry, &validate_tool_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Tool '{}' not found (needed for code-mode step '{}')",
                validate_tool_name,
                step.id
            )
        })?;

        let validate_args = serde_json::json!({
            "code": resolved_code,
        });
        let validate_result = validate_tool.execute(validate_args).await?;
        if !validate_result.success {
            anyhow::bail!(
                "validate_code failed for step '{}': {}",
                step.id,
                validate_result.error.as_deref().unwrap_or(&validate_result.output)
            );
        }

        // Extract approval_token from the validate_code response
        let validate_output: serde_json::Value =
            serde_json::from_str(&validate_result.output).unwrap_or(serde_json::Value::Null);
        let approval_token = validate_output
            .get("approval_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "validate_code for step '{}' did not return approval_token",
                    step.id
                )
            })?;

        // Step 2: execute_code
        let execute_tool_name = format!("{server}.execute_code");
        let execute_tool = find_tool(tools_registry, &execute_tool_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Tool '{}' not found (needed for code-mode step '{}')",
                execute_tool_name,
                step.id
            )
        })?;

        let execute_args = serde_json::json!({
            "code": resolved_code,
            "approval_token": approval_token,
        });
        let execute_result = execute_tool.execute(execute_args).await?;
        tool_result_to_value(&execute_result, step)
    }

    /// Tier 3: Proactive LLM prompt step.
    ///
    /// Resolves `{{...}}` templates in the prompt string, appends output from
    /// `context_from` steps as additional context, then calls the LLM and
    /// parses the response as JSON (falling back to a plain string).
    async fn execute_prompt_step(
        step: &ScriptStep,
        step_results: &HashMap<String, serde_json::Value>,
        script: &Script,
        provider: Option<&dyn Provider>,
        model: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let provider = provider
            .ok_or_else(|| anyhow::anyhow!("Prompt step '{}' requires an LLM provider", step.id))?;

        let raw_prompt = step
            .prompt
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Prompt step '{}' missing 'prompt' field", step.id))?;

        // Resolve templates in the prompt string
        let resolved_value = resolve_templates(
            &serde_json::Value::String(raw_prompt.to_string()),
            &script.variables,
            step_results,
        );
        let mut prompt = resolved_value
            .as_str()
            .unwrap_or(raw_prompt)
            .to_string();

        // Append context from referenced steps
        if let Some(context_ids) = &step.context_from {
            for ctx_id in context_ids {
                if let Some(ctx_val) = step_results.get(ctx_id.as_str()) {
                    let ctx_str = match ctx_val {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    // Truncate large contexts to keep prompt manageable
                    let truncated = if ctx_str.len() > 8000 {
                        &ctx_str[..8000]
                    } else {
                        ctx_str.as_str()
                    };
                    use std::fmt::Write;
                    let _ = write!(prompt, "\n\n--- Context from step '{ctx_id}' ---\n{truncated}");
                }
            }
        }

        let response = provider.simple_chat(&prompt, model, 0.2).await?;

        // Try to parse as JSON; fall back to plain string
        let parsed = serde_json::from_str::<serde_json::Value>(response.trim())
            .unwrap_or_else(|_| serde_json::Value::String(response.trim().to_string()));

        // Apply output_key wrapping
        if let Some(key) = &step.output_key {
            let mut map = serde_json::Map::new();
            map.insert(key.clone(), parsed);
            Ok(serde_json::Value::Object(map))
        } else {
            Ok(parsed)
        }
    }

    /// LLM-assisted recovery: ask the local LLM to diagnose and fix a failed step.
    #[allow(clippy::too_many_arguments)]
    async fn llm_assist(
        step: &ScriptStep,
        error: &str,
        step_results: &HashMap<String, serde_json::Value>,
        tools_registry: &[Box<dyn Tool>],
        provider: Option<&dyn Provider>,
        model: &str,
        script: &Script,
        deadline: tokio::time::Instant,
    ) -> Result<(serde_json::Value, String), String> {
        let provider = provider.ok_or_else(|| "No LLM provider available for llm_assist".to_string())?;

        // Gather page context if this is a browser step
        let mut page_context = String::new();
        if step
            .tool
            .as_deref()
            .map_or(false, |t| t.starts_with("browser-server."))
        {
            if let Some(dom_tool) = find_tool(tools_registry, "browser-server.get_dom") {
                if let Ok(dom_result) = dom_tool
                    .execute(serde_json::json!({}))
                    .await
                {
                    if dom_result.success {
                        // Truncate DOM to keep prompt manageable
                        let dom = &dom_result.output;
                        let truncated = if dom.len() > 4000 {
                            &dom[..4000]
                        } else {
                            dom.as_str()
                        };
                        page_context = format!("\n\nCurrent page DOM (truncated):\n{truncated}");
                    }
                }
            }
        }

        let step_json =
            serde_json::to_string_pretty(step).unwrap_or_else(|_| format!("{step:?}"));
        let prompt = format!(
            "A script step failed and needs recovery.\n\n\
             Step definition:\n{step_json}\n\n\
             Error: {error}{page_context}\n\n\
             Previous step results: {prev}\n\n\
             Suggest a corrected tool call as JSON with \"tool\" and \"args\" fields \
             (or \"code\" for code-mode steps). Return ONLY the JSON, no explanation.",
            prev = serde_json::to_string(step_results).unwrap_or_default(),
        );

        let llm_response = tokio::time::timeout_at(
            deadline,
            provider.simple_chat(&prompt, model, 0.3),
        )
        .await
        .map_err(|_| "LLM assist timed out".to_string())?
        .map_err(|e| format!("LLM call failed: {e}"))?;

        // Try to parse the LLM response as a corrected step
        let corrected: serde_json::Value =
            serde_json::from_str(llm_response.trim()).map_err(|e| {
                format!("LLM response was not valid JSON: {e}\nResponse: {llm_response}")
            })?;

        // Build a corrected step from the LLM suggestion
        let mut corrected_step = step.clone();
        if let Some(tool) = corrected.get("tool").and_then(|v| v.as_str()) {
            corrected_step.tool = Some(tool.to_string());
        }
        if let Some(args) = corrected.get("args") {
            corrected_step.args = Some(args.clone());
        }
        if let Some(code) = corrected.get("code").and_then(|v| v.as_str()) {
            corrected_step.code = Some(code.to_string());
        }

        // Retry with corrected step
        let retry_result = tokio::time::timeout_at(
            deadline,
            Self::execute_step(&corrected_step, tools_registry, step_results, script, Some(provider), model),
        )
        .await
        .map_err(|_| "Corrected step timed out".to_string())?
        .map_err(|e| format!("Corrected step failed: {e}"))?;

        Ok((retry_result, llm_response))
    }
}

// ── Template Resolution ──────────────────────────────────────────

/// Recursively resolve `{{variables.X}}` and `{{steps.ID.field}}` in a JSON value.
fn resolve_templates(
    value: &serde_json::Value,
    variables: &serde_json::Value,
    step_results: &HashMap<String, serde_json::Value>,
) -> serde_json::Value {
    match value {
        serde_json::Value::String(s) => resolve_template_string(s, variables, step_results),
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.iter()
                .map(|v| resolve_templates(v, variables, step_results))
                .collect(),
        ),
        serde_json::Value::Object(map) => {
            let mut resolved = serde_json::Map::new();
            for (k, v) in map {
                resolved.insert(k.clone(), resolve_templates(v, variables, step_results));
            }
            serde_json::Value::Object(resolved)
        }
        other => other.clone(),
    }
}

/// Resolve templates in a single string.
///
/// If the entire string is a single `{{...}}` reference that resolves to a
/// non-string JSON value, the value is returned directly (preserving type).
/// Otherwise, template placeholders are interpolated into the string.
fn resolve_template_string(
    s: &str,
    variables: &serde_json::Value,
    step_results: &HashMap<String, serde_json::Value>,
) -> serde_json::Value {
    // Fast path: no templates
    if !s.contains("{{") {
        return serde_json::Value::String(s.to_string());
    }

    // If the entire string is a single template reference, return the raw value
    let trimmed = s.trim();
    if trimmed.starts_with("{{") && trimmed.ends_with("}}") && trimmed.matches("{{").count() == 1 {
        let path = trimmed[2..trimmed.len() - 2].trim();
        if let Some(resolved) = lookup_template_path(path, variables, step_results) {
            return resolved;
        }
        // Unresolvable — return the original string
        return serde_json::Value::String(s.to_string());
    }

    // Multiple templates or mixed text — do string interpolation
    let mut result = s.to_string();
    let mut search_from = 0;
    while let Some(start) = result[search_from..].find("{{") {
        let abs_start = search_from + start;
        if let Some(end) = result[abs_start..].find("}}") {
            let abs_end = abs_start + end + 2;
            let path = result[abs_start + 2..abs_end - 2].trim();
            if let Some(value) = lookup_template_path(path, variables, step_results) {
                let replacement = match &value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                result.replace_range(abs_start..abs_end, &replacement);
                search_from = abs_start + replacement.len();
            } else {
                // Unresolvable — skip past this template
                search_from = abs_end;
            }
        } else {
            break;
        }
    }

    serde_json::Value::String(result)
}

/// Look up a dotted template path like `variables.claim_id` or `steps.open-portal.url`.
fn lookup_template_path(
    path: &str,
    variables: &serde_json::Value,
    step_results: &HashMap<String, serde_json::Value>,
) -> Option<serde_json::Value> {
    let parts: Vec<&str> = path.splitn(3, '.').collect();
    match *parts.first()? {
        "variables" => {
            if parts.len() == 2 {
                variables.get(parts[1]).cloned()
            } else {
                None
            }
        }
        "steps" => {
            if parts.len() >= 2 {
                let step_id = parts[1];
                let step_val = step_results.get(step_id)?;
                if parts.len() == 3 {
                    step_val.get(parts[2]).cloned()
                } else {
                    Some(step_val.clone())
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Convert elapsed time to milliseconds, saturating at `u64::MAX`.
fn elapsed_ms(start: &Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Find a tool by name in the registry.
fn find_tool<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> Option<&'a dyn Tool> {
    tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
}

/// Convert a `ToolResult` to a `serde_json::Value`, respecting `output_key`.
fn tool_result_to_value(result: &ToolResult, step: &ScriptStep) -> anyhow::Result<serde_json::Value> {
    if !result.success {
        anyhow::bail!(
            "Tool execution failed: {}",
            result.error.as_deref().unwrap_or("unknown error")
        );
    }

    // Try to parse the output as JSON
    let parsed = serde_json::from_str::<serde_json::Value>(&result.output)
        .unwrap_or_else(|_| serde_json::Value::String(result.output.clone()));

    // If output_key is specified, wrap in an object
    if let Some(key) = &step.output_key {
        let mut map = serde_json::Map::new();
        map.insert(key.clone(), parsed);
        Ok(serde_json::Value::Object(map))
    } else {
        Ok(parsed)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    // ── Detection tests ──────────────────────────────────────────

    #[test]
    fn detect_script_from_wrapped_payload() {
        let input = r#"{
            "script": {
                "version": "1.0",
                "name": "test",
                "steps": [{"id": "s1", "tool": "browser-server.navigate", "args": {"url": "https://example.com"}}]
            },
            "parameters": {"foo": "bar"}
        }"#;

        let script = ScriptEngine::detect(input).unwrap();
        assert_eq!(script.name, "test");
        assert_eq!(script.steps.len(), 1);
        assert_eq!(script.steps[0].id, "s1");
    }

    #[test]
    fn detect_script_direct() {
        let input = r##"{
            "version": "1.0",
            "name": "direct-test",
            "steps": [
                {"id": "step1", "tool": "browser-server.click", "args": {"selector": "#btn"}}
            ]
        }"##;

        let script = ScriptEngine::detect(input).unwrap();
        assert_eq!(script.name, "direct-test");
        assert_eq!(script.steps.len(), 1);
    }

    #[test]
    fn detect_returns_none_for_chat() {
        assert!(ScriptEngine::detect("Hello, how are you?").is_none());
    }

    #[test]
    fn detect_returns_none_for_json_without_script() {
        let input = r#"{"message": "hi", "data": 42}"#;
        assert!(ScriptEngine::detect(input).is_none());
    }

    #[test]
    fn detect_returns_none_for_json_with_version_but_no_steps() {
        let input = r#"{"version": "2.0", "data": "test"}"#;
        assert!(ScriptEngine::detect(input).is_none());
    }

    // ── Template resolution tests ────────────────────────────────

    #[test]
    fn template_resolution_variables() {
        let variables = serde_json::json!({"claim_id": "CLM-12345"});
        let step_results = HashMap::new();
        let input = serde_json::json!("{{variables.claim_id}}");

        let resolved = resolve_templates(&input, &variables, &step_results);
        assert_eq!(resolved, serde_json::json!("CLM-12345"));
    }

    #[test]
    fn template_resolution_steps() {
        let variables = serde_json::json!({});
        let mut step_results = HashMap::new();
        step_results.insert(
            "step1".to_string(),
            serde_json::json!({"status": "approved", "code": 200}),
        );

        let input = serde_json::json!("{{steps.step1.status}}");
        let resolved = resolve_templates(&input, &variables, &step_results);
        assert_eq!(resolved, serde_json::json!("approved"));
    }

    #[test]
    fn template_resolution_preserves_non_string_types() {
        let variables = serde_json::json!({});
        let mut step_results = HashMap::new();
        step_results.insert("step1".to_string(), serde_json::json!({"count": 42}));

        let input = serde_json::json!("{{steps.step1.count}}");
        let resolved = resolve_templates(&input, &variables, &step_results);
        assert_eq!(resolved, serde_json::json!(42));
    }

    #[test]
    fn template_resolution_nested() {
        let variables = serde_json::json!({"url": "https://example.com"});
        let step_results = HashMap::new();
        let input = serde_json::json!({
            "navigate": {"url": "{{variables.url}}"},
            "items": ["{{variables.url}}", "static"]
        });

        let resolved = resolve_templates(&input, &variables, &step_results);
        assert_eq!(
            resolved,
            serde_json::json!({
                "navigate": {"url": "https://example.com"},
                "items": ["https://example.com", "static"]
            })
        );
    }

    #[test]
    fn template_missing_variable_preserved() {
        let variables = serde_json::json!({});
        let step_results = HashMap::new();
        let input = serde_json::json!("{{variables.missing}}");

        let resolved = resolve_templates(&input, &variables, &step_results);
        assert_eq!(resolved, serde_json::json!("{{variables.missing}}"));
    }

    #[test]
    fn template_mixed_text_and_templates() {
        let variables = serde_json::json!({"name": "Alice"});
        let step_results = HashMap::new();
        let input = serde_json::json!("Hello {{variables.name}}, welcome!");

        let resolved = resolve_templates(&input, &variables, &step_results);
        assert_eq!(resolved, serde_json::json!("Hello Alice, welcome!"));
    }

    #[test]
    fn template_whole_step_result() {
        let variables = serde_json::json!({});
        let mut step_results = HashMap::new();
        step_results.insert(
            "extract".to_string(),
            serde_json::json!({"a": 1, "b": 2}),
        );

        let input = serde_json::json!("{{steps.extract}}");
        let resolved = resolve_templates(&input, &variables, &step_results);
        assert_eq!(resolved, serde_json::json!({"a": 1, "b": 2}));
    }

    // ── Deserialization tests ────────────────────────────────────

    #[test]
    fn script_step_deserialization_tool_mode() {
        let json = r#"{
            "id": "code1",
            "tool_mode": "code",
            "server": "browser-server",
            "code": "return 42;"
        }"#;

        let step: ScriptStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.id, "code1");
        assert_eq!(step.tool_mode.as_deref(), Some("code"));
        assert_eq!(step.server.as_deref(), Some("browser-server"));
        assert_eq!(step.code.as_deref(), Some("return 42;"));
        assert!(step.tool.is_none());
        assert_eq!(step.timeout_secs, 60); // default
    }

    #[test]
    fn script_step_deserialization_tool() {
        let json = r#"{
            "id": "nav",
            "tool": "browser-server.navigate",
            "args": {"url": "https://example.com"},
            "timeout_secs": 30
        }"#;

        let step: ScriptStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.id, "nav");
        assert_eq!(step.tool.as_deref(), Some("browser-server.navigate"));
        assert_eq!(step.timeout_secs, 30);
        assert!(step.tool_mode.is_none());
    }

    #[test]
    fn failure_strategy_deserialization() {
        let cases = [
            (r#""abort""#, "Abort"),
            (r#""skip""#, "Skip"),
            (r#""retry""#, "Retry"),
            (r#""llm_assist""#, "LlmAssist"),
        ];

        for (json, expected_debug) in &cases {
            let strategy: FailureStrategy = serde_json::from_str(json).unwrap();
            assert!(
                format!("{strategy:?}").contains(expected_debug),
                "Expected {:?} to contain {expected_debug}",
                strategy
            );
        }
    }

    #[test]
    fn output_template_resolution() {
        let variables = serde_json::json!({"claim_id": "CLM-999"});
        let mut step_results = HashMap::new();
        step_results.insert(
            "extract".to_string(),
            serde_json::json!({"status": "denied"}),
        );
        step_results.insert(
            "screenshot".to_string(),
            serde_json::json!({"screenshot_data": "base64..."}),
        );

        let mut output_template = serde_json::Map::new();
        output_template.insert(
            "claim_status".to_string(),
            serde_json::json!("{{steps.extract.status}}"),
        );
        output_template.insert(
            "screenshot".to_string(),
            serde_json::json!("{{steps.screenshot.screenshot_data}}"),
        );
        output_template.insert(
            "claim_id".to_string(),
            serde_json::json!("{{variables.claim_id}}"),
        );

        let mut resolved = serde_json::Map::new();
        for (key, tmpl) in &output_template {
            resolved.insert(
                key.clone(),
                resolve_templates(tmpl, &variables, &step_results),
            );
        }

        assert_eq!(resolved["claim_status"], serde_json::json!("denied"));
        assert_eq!(resolved["screenshot"], serde_json::json!("base64..."));
        assert_eq!(resolved["claim_id"], serde_json::json!("CLM-999"));
    }

    // ── Integration test with mock tools ─────────────────────────

    struct MockTool {
        tool_name: String,
        response: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            "mock"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                success: true,
                output: self.response.clone(),
                error: None,
            })
        }
    }

    struct FailingTool {
        tool_name: String,
    }

    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            "always fails"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("intentional failure".into()),
            })
        }
    }

    fn default_config() -> ScriptEngineConfig {
        ScriptEngineConfig::default()
    }

    struct MockProvider {
        response: String,
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn execute_prompt_step_json_response() {
        let provider = MockProvider {
            response: r##"{"selector": "#addr-3", "matched_address": "42 Test St"}"##.into(),
        };
        let tools: Vec<Box<dyn Tool>> = vec![];

        let script = Script {
            version: "1.0".into(),
            name: "prompt-test".into(),
            variables: serde_json::json!({"address": "42 Test Street"}),
            steps: vec![ScriptStep {
                id: "select".into(),
                tool: None,
                args: None,
                tool_mode: None,
                server: None,
                code: None,
                prompt: Some("Find the address matching {{variables.address}}".into()),
                context_from: None,
                timeout_secs: 30,
                output_key: None,
                on_failure: None,
            }],
            output: None,
        };

        let observer = crate::observability::NoopObserver;
        let config = default_config();
        let result = ScriptEngine::execute(
            &script,
            &tools,
            Some(&provider),
            "test-model",
            &observer,
            &config,
        )
        .await;

        assert!(result.success);
        assert_eq!(result.execution_details.steps_succeeded, 1);
        let step_output = result.output.get("select").unwrap();
        assert_eq!(step_output["selector"], "#addr-3");
        assert_eq!(step_output["matched_address"], "42 Test St");
    }

    #[tokio::test]
    async fn execute_prompt_step_with_context_from() {
        let provider = MockProvider {
            response: r#"{"speed": "80Mbps", "price": "£29.99"}"#.into(),
        };
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            tool_name: "browser-server.get_dom".into(),
            response: r#"{"dom": "<div>broadband packages here</div>"}"#.into(),
        })];

        let script = Script {
            version: "1.0".into(),
            name: "context-test".into(),
            variables: serde_json::json!({}),
            steps: vec![
                ScriptStep {
                    id: "get-dom".into(),
                    tool: Some("browser-server.get_dom".into()),
                    args: None,
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 30,
                    output_key: None,
                    on_failure: None,
                },
                ScriptStep {
                    id: "extract".into(),
                    tool: None,
                    args: None,
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: Some("Extract broadband data from the DOM".into()),
                    context_from: Some(vec!["get-dom".into()]),
                    timeout_secs: 30,
                    output_key: None,
                    on_failure: None,
                },
            ],
            output: None,
        };

        let observer = crate::observability::NoopObserver;
        let config = default_config();
        let result = ScriptEngine::execute(
            &script,
            &tools,
            Some(&provider),
            "test-model",
            &observer,
            &config,
        )
        .await;

        assert!(result.success);
        assert_eq!(result.execution_details.steps_succeeded, 2);
        let extract_output = result.output.get("extract").unwrap();
        assert_eq!(extract_output["speed"], "80Mbps");
        assert_eq!(extract_output["price"], "£29.99");
    }

    #[tokio::test]
    async fn execute_prompt_step_plain_text_fallback() {
        let provider = MockProvider {
            response: "This is not valid JSON, just plain text".into(),
        };
        let tools: Vec<Box<dyn Tool>> = vec![];

        let script = Script {
            version: "1.0".into(),
            name: "plain-text-test".into(),
            variables: serde_json::json!({}),
            steps: vec![ScriptStep {
                id: "ask".into(),
                tool: None,
                args: None,
                tool_mode: None,
                server: None,
                code: None,
                prompt: Some("What is the meaning of life?".into()),
                context_from: None,
                timeout_secs: 30,
                output_key: None,
                on_failure: None,
            }],
            output: None,
        };

        let observer = crate::observability::NoopObserver;
        let config = default_config();
        let result = ScriptEngine::execute(
            &script,
            &tools,
            Some(&provider),
            "test-model",
            &observer,
            &config,
        )
        .await;

        assert!(result.success);
        let step_output = result.output.get("ask").unwrap();
        assert_eq!(
            step_output.as_str().unwrap(),
            "This is not valid JSON, just plain text"
        );
    }

    #[tokio::test]
    async fn execute_prompt_step_no_provider_fails() {
        let tools: Vec<Box<dyn Tool>> = vec![];

        let script = Script {
            version: "1.0".into(),
            name: "no-provider-test".into(),
            variables: serde_json::json!({}),
            steps: vec![ScriptStep {
                id: "ask".into(),
                tool: None,
                args: None,
                tool_mode: None,
                server: None,
                code: None,
                prompt: Some("Hello".into()),
                context_from: None,
                timeout_secs: 30,
                output_key: None,
                on_failure: None,
            }],
            output: None,
        };

        let observer = crate::observability::NoopObserver;
        let config = default_config();
        let result =
            ScriptEngine::execute(&script, &tools, None, "test-model", &observer, &config).await;

        assert!(!result.success);
        assert!(result.error.unwrap().contains("requires an LLM provider"));
    }

    #[tokio::test]
    async fn execute_chained_tool_steps() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(MockTool {
                tool_name: "browser-server.navigate".into(),
                response: r#"{"success": true, "url": "https://example.com"}"#.into(),
            }),
            Box::new(MockTool {
                tool_name: "browser-server.get_text".into(),
                response: r#"{"text": "Example Domain"}"#.into(),
            }),
            Box::new(MockTool {
                tool_name: "browser-server.screenshot".into(),
                response: r#"{"data": "base64png"}"#.into(),
            }),
        ];

        let script = Script {
            version: "1.0".into(),
            name: "test-chain".into(),
            variables: serde_json::json!({"url": "https://example.com"}),
            steps: vec![
                ScriptStep {
                    id: "nav".into(),
                    tool: Some("browser-server.navigate".into()),
                    args: Some(serde_json::json!({"url": "{{variables.url}}"})),
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 30,
                    output_key: None,
                    on_failure: None,
                },
                ScriptStep {
                    id: "text".into(),
                    tool: Some("browser-server.get_text".into()),
                    args: Some(serde_json::json!({"selector": "h1"})),
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 30,
                    output_key: None,
                    on_failure: None,
                },
                ScriptStep {
                    id: "shot".into(),
                    tool: Some("browser-server.screenshot".into()),
                    args: Some(serde_json::json!({"full_page": true})),
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 30,
                    output_key: None,
                    on_failure: None,
                },
            ],
            output: Some({
                let mut m = serde_json::Map::new();
                m.insert(
                    "heading".into(),
                    serde_json::json!("{{steps.text.text}}"),
                );
                m.insert(
                    "screenshot".into(),
                    serde_json::json!("{{steps.shot.data}}"),
                );
                m
            }),
        };

        let observer = crate::observability::NoopObserver;
        let config = default_config();

        let result =
            ScriptEngine::execute(&script, &tools, None, "test-model", &observer, &config).await;

        assert!(result.success);
        assert_eq!(result.execution_details.steps_executed, 3);
        assert_eq!(result.execution_details.steps_succeeded, 3);
        assert_eq!(result.execution_details.steps_failed, 0);
        assert_eq!(result.output["heading"], "Example Domain");
        assert_eq!(result.output["screenshot"], "base64png");
    }

    #[tokio::test]
    async fn execute_abort_on_failure() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(MockTool {
                tool_name: "step-a".into(),
                response: r#"{"ok": true}"#.into(),
            }),
            Box::new(FailingTool {
                tool_name: "step-b".into(),
            }),
            Box::new(MockTool {
                tool_name: "step-c".into(),
                response: r#"{"ok": true}"#.into(),
            }),
        ];

        let script = Script {
            version: "1.0".into(),
            name: "abort-test".into(),
            variables: serde_json::json!({}),
            steps: vec![
                ScriptStep {
                    id: "a".into(),
                    tool: Some("step-a".into()),
                    args: None,
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 10,
                    output_key: None,
                    on_failure: None, // default = Abort
                },
                ScriptStep {
                    id: "b".into(),
                    tool: Some("step-b".into()),
                    args: None,
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 10,
                    output_key: None,
                    on_failure: None,
                },
                ScriptStep {
                    id: "c".into(),
                    tool: Some("step-c".into()),
                    args: None,
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 10,
                    output_key: None,
                    on_failure: None,
                },
            ],
            output: None,
        };

        let observer = crate::observability::NoopObserver;
        let config = default_config();
        let result =
            ScriptEngine::execute(&script, &tools, None, "test-model", &observer, &config).await;

        assert!(!result.success);
        assert_eq!(result.execution_details.steps_succeeded, 1);
        assert_eq!(result.execution_details.steps_failed, 1);
        assert_eq!(result.execution_details.steps_executed, 2);
        let err = result.error.as_deref().unwrap();
        assert!(
            err.contains("'b' failed"),
            "Expected error about step 'b', got: {err}"
        );
    }

    #[tokio::test]
    async fn execute_skip_on_failure() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(FailingTool {
                tool_name: "flaky".into(),
            }),
            Box::new(MockTool {
                tool_name: "solid".into(),
                response: r#"{"result": "ok"}"#.into(),
            }),
        ];

        let script = Script {
            version: "1.0".into(),
            name: "skip-test".into(),
            variables: serde_json::json!({}),
            steps: vec![
                ScriptStep {
                    id: "flaky-step".into(),
                    tool: Some("flaky".into()),
                    args: None,
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 10,
                    output_key: None,
                    on_failure: Some(FailureStrategy::Skip),
                },
                ScriptStep {
                    id: "solid-step".into(),
                    tool: Some("solid".into()),
                    args: None,
                    tool_mode: None,
                    server: None,
                    code: None,
                    prompt: None,
                    context_from: None,
                    timeout_secs: 10,
                    output_key: None,
                    on_failure: None,
                },
            ],
            output: None,
        };

        let observer = crate::observability::NoopObserver;
        let config = default_config();
        let result =
            ScriptEngine::execute(&script, &tools, None, "test-model", &observer, &config).await;

        assert!(result.success);
        assert_eq!(result.execution_details.steps_succeeded, 1);
        assert_eq!(result.execution_details.steps_failed, 1);
        assert_eq!(result.execution_details.steps_executed, 2);
    }

    #[test]
    fn find_tool_by_name() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(MockTool {
                tool_name: "a.b".into(),
                response: "{}".into(),
            }),
            Box::new(MockTool {
                tool_name: "c.d".into(),
                response: "{}".into(),
            }),
        ];

        assert!(find_tool(&tools, "a.b").is_some());
        assert!(find_tool(&tools, "c.d").is_some());
        assert!(find_tool(&tools, "e.f").is_none());
    }
}
