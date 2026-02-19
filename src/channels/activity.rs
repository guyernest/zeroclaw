use super::traits::{Channel, ChannelMessage};
use crate::config::ActivityChannelConfig;
use async_trait::async_trait;
use std::sync::Mutex;
use std::time::Duration;
use uuid::Uuid;

/// AWS Step Functions Activity channel — polls `GetActivityTask` and replies
/// via `SendTaskSuccess`, bridging SFN activities into `ZeroClaw`'s channel bus.
pub struct ActivityChannel {
    activity_arn: String,
    worker_name: String,
    client: aws_sdk_sfn::Client,
    poll_interval: Duration,
    heartbeat_interval: Duration,
    heartbeat_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl ActivityChannel {
    /// Create a new `ActivityChannel` from config.
    ///
    /// Loads AWS credentials from the environment / profile and builds the SFN client.
    pub async fn new(config: &ActivityChannelConfig) -> anyhow::Result<Self> {
        let mut aws_builder = aws_config::defaults(aws_config::BehaviorVersion::latest());

        if let Some(ref profile) = config.aws_profile {
            aws_builder = aws_builder.profile_name(profile);
        }
        if let Some(ref region) = config.aws_region {
            aws_builder = aws_builder.region(aws_sdk_sfn::config::Region::new(region.to_owned()));
        }

        let aws_config = aws_builder.load().await;
        let client = aws_sdk_sfn::Client::new(&aws_config);

        Ok(Self {
            activity_arn: config.activity_arn.clone(),
            worker_name: config.worker_name.clone(),
            client,
            poll_interval: Duration::from_millis(config.poll_interval_ms),
            heartbeat_interval: Duration::from_secs(config.heartbeat_interval_secs),
            heartbeat_handle: Mutex::new(None),
        })
    }

    /// Build the JSON output envelope sent back to Step Functions.
    pub fn format_output(message: &str) -> String {
        serde_json::json!({
            "output": message,
            "success": true,
            "approved": true,
        })
        .to_string()
    }
}

#[async_trait]
impl Channel for ActivityChannel {
    fn name(&self) -> &str {
        "activity"
    }

    async fn send(&self, message: &str, recipient: &str) -> anyhow::Result<()> {
        let output = Self::format_output(message);
        self.client
            .send_task_success()
            .task_token(recipient)
            .output(&output)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("SendTaskSuccess failed: {e}"))?;

        tracing::info!(
            "Activity task completed (token prefix: {}...)",
            &recipient[..recipient.len().min(12)]
        );
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        tracing::info!(
            "Activity channel listening on {} as worker '{}'",
            self.activity_arn,
            self.worker_name
        );

        loop {
            if tx.is_closed() {
                return Ok(());
            }

            let result = self
                .client
                .get_activity_task()
                .activity_arn(&self.activity_arn)
                .worker_name(&self.worker_name)
                .send()
                .await;

            match result {
                Ok(output) => {
                    let task_token = output.task_token().unwrap_or_default();
                    let input = output.input().unwrap_or_default();

                    if task_token.is_empty() {
                        // No task available — sleep and retry
                        tokio::time::sleep(self.poll_interval).await;
                        continue;
                    }

                    tracing::info!(
                        "Activity task received (token prefix: {}...)",
                        &task_token[..task_token.len().min(12)]
                    );

                    let msg = ChannelMessage {
                        id: Uuid::new_v4().to_string(),
                        sender: task_token.to_string(),
                        content: input.to_string(),
                        channel: "activity".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    };

                    if tx.send(msg).await.is_err() {
                        return Ok(());
                    }
                }
                Err(e) => {
                    tracing::warn!("GetActivityTask error: {e:#}");
                    tokio::time::sleep(self.poll_interval).await;
                }
            }
        }
    }

    async fn health_check(&self) -> bool {
        // Verify we can describe the activity (validates ARN + credentials)
        self.client
            .describe_activity()
            .activity_arn(&self.activity_arn)
            .send()
            .await
            .is_ok()
    }

    async fn start_typing(&self, task_token: &str) -> anyhow::Result<()> {
        let client = self.client.clone();
        let token = task_token.to_string();
        let interval = self.heartbeat_interval;

        let handle = tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            loop {
                tick.tick().await;
                if let Err(e) = client.send_task_heartbeat().task_token(&token).send().await {
                    tracing::warn!("Activity heartbeat failed: {e}");
                    break;
                }
            }
        });

        if let Ok(mut guard) = self.heartbeat_handle.lock() {
            if let Some(old) = guard.take() {
                old.abort();
            }
            *guard = Some(handle);
        }

        Ok(())
    }

    async fn stop_typing(&self, _task_token: &str) -> anyhow::Result<()> {
        if let Ok(mut guard) = self.heartbeat_handle.lock() {
            if let Some(handle) = guard.take() {
                handle.abort();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_format_output_structure() {
        let output = ActivityChannel::format_output("Hello, world!");
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["output"], "Hello, world!");
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["approved"], true);
    }

    #[test]
    fn channel_format_output_empty_message() {
        let output = ActivityChannel::format_output("");
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["output"], "");
        assert_eq!(parsed["success"], true);
    }

    #[test]
    fn channel_format_output_with_special_chars() {
        let output = ActivityChannel::format_output("Line 1\nLine 2\t\"quoted\"");
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["output"], "Line 1\nLine 2\t\"quoted\"");
    }

    #[test]
    fn channel_format_output_is_valid_json() {
        let output = ActivityChannel::format_output("test");
        // Must be valid JSON that Step Functions can accept
        assert!(serde_json::from_str::<serde_json::Value>(&output).is_ok());
        assert!(output.contains("\"output\""));
        assert!(output.contains("\"success\""));
        assert!(output.contains("\"approved\""));
    }
}
