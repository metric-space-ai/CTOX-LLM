//! Pinned Qwen3.8 tokenizer and text-only chat/tool template.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{EngineError, Result};

pub const TOKENIZER_SHA256: &str =
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3";
pub const CHAT_TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
pub const END_OF_TEXT_ID: u32 = 248_044;
pub const IM_START_ID: u32 = 248_045;
pub const IM_END_ID: u32 = 248_046;
pub const THINK_START_ID: u32 = 248_068;
pub const THINK_END_ID: u32 = 248_069;
pub const TOKENIZER_VOCAB_SIZE: usize = 248_077;

const XHIGH_REASONING: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW_REASONING: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
const TOOL_PREAMBLE: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug)]
pub struct Qwen38Tokenizer {
    tokenizer: Tokenizer,
    sha256: String,
    chat_template_sha256: String,
}

impl Qwen38Tokenizer {
    pub fn from_release_files(
        tokenizer_path: impl AsRef<Path>,
        chat_template_path: impl AsRef<Path>,
        expected_tokenizer_sha256: &str,
        expected_chat_template_sha256: &str,
    ) -> Result<Self> {
        require_sha256(expected_tokenizer_sha256, "tokenizer")?;
        require_sha256(expected_chat_template_sha256, "chat template")?;
        let encoded = fs::read(tokenizer_path)?;
        let chat_template = fs::read(chat_template_path)?;
        Self::from_release_bytes(
            encoded,
            chat_template,
            expected_tokenizer_sha256,
            expected_chat_template_sha256,
        )
    }

    pub fn from_release_bytes(
        encoded: Vec<u8>,
        chat_template: Vec<u8>,
        expected_tokenizer_sha256: &str,
        expected_chat_template_sha256: &str,
    ) -> Result<Self> {
        require_sha256(expected_tokenizer_sha256, "tokenizer")?;
        require_sha256(expected_chat_template_sha256, "chat template")?;
        let sha256 = format!("{:x}", Sha256::digest(&encoded));
        if sha256 != expected_tokenizer_sha256 || sha256 != TOKENIZER_SHA256 {
            return Err(EngineError::InvalidArtifact(format!(
                "tokenizer SHA-256 differs: expected {expected_tokenizer_sha256}, pinned {TOKENIZER_SHA256}, observed {sha256}"
            )));
        }
        let chat_template_sha256 = format!("{:x}", Sha256::digest(&chat_template));
        if chat_template_sha256 != expected_chat_template_sha256
            || chat_template_sha256 != CHAT_TEMPLATE_SHA256
        {
            return Err(EngineError::InvalidArtifact(format!(
                "chat-template SHA-256 differs: expected {expected_chat_template_sha256}, pinned {CHAT_TEMPLATE_SHA256}, observed {chat_template_sha256}"
            )));
        }
        let tokenizer = Tokenizer::from_bytes(&encoded).map_err(tokenizer_error)?;
        if tokenizer.get_vocab_size(true) != TOKENIZER_VOCAB_SIZE
            || tokenizer.token_to_id("<|endoftext|>") != Some(END_OF_TEXT_ID)
            || tokenizer.token_to_id("<|im_start|>") != Some(IM_START_ID)
            || tokenizer.token_to_id("<|im_end|>") != Some(IM_END_ID)
            || tokenizer.token_to_id("<think>") != Some(THINK_START_ID)
            || tokenizer.token_to_id("</think>") != Some(THINK_END_ID)
        {
            return Err(EngineError::InvalidArtifact(
                "tokenizer vocabulary or Qwen special-token IDs differ".into(),
            ));
        }
        Ok(Self {
            tokenizer,
            sha256,
            chat_template_sha256,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn chat_template_sha256(&self) -> &str {
        &self.chat_template_sha256
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(tokenizer_error)?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(tokenizer_error)
    }

    pub fn render_and_encode(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<(String, Vec<u32>)> {
        let rendered = render_chat(messages, options)?;
        let ids = self.encode(&rendered)?;
        Ok((rendered, ids))
    }
}

fn tokenizer_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidArtifact(format!("Qwen tokenizer failed: {error}"))
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(EngineError::InvalidArtifact(format!(
            "{label} digest is not lowercase SHA-256"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningEffort {
    #[default]
    XHigh,
    Medium,
    Low,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatTemplateOptions {
    pub tools: Vec<Value>,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub reasoning_effort: ReasoningEffort,
    pub preserve_thinking: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            add_generation_prompt: true,
            enable_thinking: true,
            reasoning_effort: ReasoningEffort::XHigh,
            preserve_thinking: true,
        }
    }
}

pub fn render_chat(messages: &[ChatMessage], options: &ChatTemplateOptions) -> Result<String> {
    if messages.is_empty() {
        return template_error("no messages provided");
    }
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System && index != 0 {
            return template_error("system message must be at the beginning");
        }
        if message.role != ChatRole::Assistant && !message.tool_calls.is_empty() {
            return template_error("only assistant messages may contain tool calls");
        }
        if message
            .tool_calls
            .iter()
            .any(|call| call.name.trim().is_empty())
        {
            return template_error("tool call name is empty");
        }
    }
    if options.tools.iter().any(|tool| !tool.is_object()) {
        return template_error("tool definitions must be JSON objects");
    }

    let reasoning = if options.enable_thinking {
        match options.reasoning_effort {
            ReasoningEffort::XHigh => XHIGH_REASONING,
            ReasoningEffort::Medium => "",
            ReasoningEffort::Low => LOW_REASONING,
        }
    } else {
        ""
    };
    let first_is_system = messages[0].role == ChatRole::System;
    let mut output = String::new();
    if !options.tools.is_empty() {
        output.push_str("<|im_start|>system\n");
        if !reasoning.is_empty() {
            output.push_str(reasoning);
            output.push_str("\n\n");
        }
        output.push_str(TOOL_PREAMBLE);
        for tool in &options.tools {
            output.push('\n');
            output.push_str(&jinja_json(tool)?);
        }
        output.push_str("\n</tools>");
        output.push_str(TOOL_INSTRUCTIONS);
        if first_is_system {
            let system = content(messages.first().expect("messages are non-empty"));
            if !system.is_empty() {
                output.push_str("\n\n");
                output.push_str(system);
            }
        }
        output.push_str("<|im_end|>\n");
    } else if first_is_system {
        let system = content(&messages[0]);
        if !system.is_empty() {
            output.push_str("<|im_start|>system\n");
            if !reasoning.is_empty() {
                output.push_str(reasoning);
                output.push_str("\n\n");
            }
            output.push_str(system);
            output.push_str("<|im_end|>\n");
        } else if !reasoning.is_empty() {
            output.push_str("<|im_start|>system\n");
            output.push_str(reasoning);
            output.push_str("<|im_end|>\n");
        }
    } else if !reasoning.is_empty() {
        output.push_str("<|im_start|>system\n");
        output.push_str(reasoning);
        output.push_str("<|im_end|>\n");
    }

    let last_query_index = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| {
            message.role == ChatRole::User
                && !(content(message).starts_with("<tool_response>")
                    && content(message).ends_with("</tool_response>"))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| EngineError::InvalidArtifact("no user query found in messages".into()))?;

    for (index, message) in messages.iter().enumerate() {
        let message_content = content(message);
        match message.role {
            ChatRole::System => {}
            ChatRole::User => {
                output.push_str("<|im_start|>user\n");
                output.push_str(message_content);
                output.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                output.push_str("<|im_start|>assistant\n");
                if options.preserve_thinking || index > last_query_index {
                    output.push_str("<think>\n");
                    output.push_str(
                        message
                            .reasoning_content
                            .as_deref()
                            .unwrap_or_default()
                            .trim(),
                    );
                    output.push_str("\n</think>\n\n");
                }
                output.push_str(message_content);
                for (tool_index, call) in message.tool_calls.iter().enumerate() {
                    if tool_index == 0 && !message_content.is_empty() {
                        output.push_str("\n\n");
                    } else if tool_index > 0 {
                        output.push('\n');
                    }
                    output.push_str("<tool_call>\n<function=");
                    output.push_str(&call.name);
                    output.push_str(">\n");
                    for (name, value) in &call.arguments {
                        output.push_str("<parameter=");
                        output.push_str(name);
                        output.push_str(">\n");
                        if let Some(value) = value.as_str() {
                            output.push_str(value);
                        } else {
                            output.push_str(&jinja_json(value)?);
                        }
                        output.push_str("\n</parameter>\n");
                    }
                    output.push_str("</function>\n</tool_call>");
                }
                output.push_str("<|im_end|>\n");
            }
            ChatRole::Tool => {
                if index == 0 || messages[index - 1].role != ChatRole::Tool {
                    output.push_str("<|im_start|>user");
                }
                output.push_str("\n<tool_response>\n");
                output.push_str(message_content);
                output.push_str("\n</tool_response>");
                if index + 1 == messages.len() || messages[index + 1].role != ChatRole::Tool {
                    output.push_str("<|im_end|>\n");
                }
            }
        }
    }
    if options.add_generation_prompt {
        output.push_str("<|im_start|>assistant\n<think>\n");
        if !options.enable_thinking {
            output.push_str("\n</think>\n\n");
        }
    }
    Ok(output)
}

fn content(message: &ChatMessage) -> &str {
    message.content.as_deref().unwrap_or_default().trim()
}

fn template_error<T>(message: &str) -> Result<T> {
    Err(EngineError::InvalidArtifact(format!(
        "Qwen chat template failed: {message}"
    )))
}

fn jinja_json(value: &Value) -> Result<String> {
    let compact = serde_json::to_string(value)?;
    let mut output = String::with_capacity(compact.len() + compact.len() / 8);
    let mut escaped = false;
    let mut string = false;
    for character in compact.chars() {
        if string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                string = false;
            }
            continue;
        }
        match character {
            '"' => {
                string = true;
                output.push(character);
            }
            ':' | ',' => {
                output.push(character);
                output.push(' ');
            }
            _ => output.push(character),
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_pinned_xhigh_and_disabled_thinking_prompts() {
        let messages = [ChatMessage::text(ChatRole::User, "Hallo Welt!")];
        let rendered = render_chat(&messages, &ChatTemplateOptions::default()).unwrap();
        assert_eq!(
            rendered,
            format!(
                "<|im_start|>system\n{XHIGH_REASONING}<|im_end|>\n<|im_start|>user\nHallo Welt!<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
        let options = ChatTemplateOptions {
            enable_thinking: false,
            ..ChatTemplateOptions::default()
        };
        assert_eq!(
            render_chat(&[ChatMessage::text(ChatRole::User, "2+2?")], &options).unwrap(),
            "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn renders_low_reasoning_system_message_exactly() {
        let options = ChatTemplateOptions {
            reasoning_effort: ReasoningEffort::Low,
            ..ChatTemplateOptions::default()
        };
        let messages = [
            ChatMessage::text(ChatRole::System, " Du bist präzise. "),
            ChatMessage::text(ChatRole::User, "Antworte kurz."),
        ];
        assert_eq!(
            render_chat(&messages, &options).unwrap(),
            format!(
                "<|im_start|>system\n{LOW_REASONING}\n\nDu bist präzise.<|im_end|>\n<|im_start|>user\nAntworte kurz.<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
    }

    #[test]
    fn renders_tool_calls_and_consecutive_responses() {
        let mut arguments = Map::new();
        arguments.insert("city".into(), Value::String("Berlin".into()));
        let messages = [
            ChatMessage::text(ChatRole::User, "Wetter?"),
            ChatMessage {
                role: ChatRole::Assistant,
                content: Some(String::new()),
                reasoning_content: None,
                tool_calls: vec![ToolCall {
                    name: "weather".into(),
                    arguments,
                }],
            },
            ChatMessage::text(ChatRole::Tool, "18 C"),
            ChatMessage::text(ChatRole::Assistant, "Es sind 18 C."),
        ];
        let options = ChatTemplateOptions {
            tools: vec![json!({
                "type": "function",
                "function": {"name": "weather", "description": "Get weather"}
            })],
            add_generation_prompt: false,
            ..ChatTemplateOptions::default()
        };
        let rendered = render_chat(&messages, &options).unwrap();
        assert!(rendered.contains(
            "<tool_call>\n<function=weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>"
        ));
        assert!(rendered
            .contains("<|im_start|>user\n<tool_response>\n18 C\n</tool_response><|im_end|>\n"));
        assert!(rendered.ends_with("Es sind 18 C.<|im_end|>\n"));
    }

    #[test]
    fn rejects_invalid_message_topology() {
        assert!(render_chat(&[], &ChatTemplateOptions::default()).is_err());
        assert!(render_chat(
            &[
                ChatMessage::text(ChatRole::User, "x"),
                ChatMessage::text(ChatRole::System, "late"),
            ],
            &ChatTemplateOptions::default(),
        )
        .is_err());
        assert!(render_chat(
            &[ChatMessage::text(ChatRole::Assistant, "no user")],
            &ChatTemplateOptions::default(),
        )
        .is_err());
    }
}
