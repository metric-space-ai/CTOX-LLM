use std::path::PathBuf;

use clap::Parser;
use ctox_qwen38_27b::tokenizer::{
    render_chat, ChatMessage, ChatRole, ChatTemplateOptions, Qwen38Tokenizer, ReasoningEffort,
    ToolCall, CHAT_TEMPLATE_SHA256, TOKENIZER_SHA256,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify the pinned Qwen3.8 tokenizer and text chat template")]
struct Args {
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    chat_template: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let tokenizer = Qwen38Tokenizer::from_release_files(
        args.tokenizer,
        args.chat_template,
        TOKENIZER_SHA256,
        CHAT_TEMPLATE_SHA256,
    )?;
    let text_cases: &[(&str, &[u32])] = &[
        ("Hello", &[9_419]),
        ("Hallo Welt!", &[75_403, 43_466, 0]),
        ("こんにちは世界", &[85_951, 96_748]),
        (
            "مرحبا بالعالم",
            &[148_739, 28_850, 150_027, 182_946, 149_650],
        ),
        (
            "fn main() { println!(\"hi\"); }",
            &[8_556, 1_822, 363, 313, 13_356, 16_715, 5_834, 4_876, 333],
        ),
        ("🙂 café", &[169_171, 50_203]),
    ];
    for (text, expected) in text_cases {
        let observed = tokenizer.encode(text)?;
        if observed != *expected {
            anyhow::bail!("tokenizer golden differs for {text:?}: {observed:?}");
        }
        if tokenizer.decode(&observed, false)? != *text {
            anyhow::bail!("tokenizer round trip differs for {text:?}");
        }
    }

    let user = [ChatMessage::text(ChatRole::User, "Hallo Welt!")];
    let (rendered, ids) = tokenizer.render_and_encode(&user, &ChatTemplateOptions::default())?;
    let expected_ids = [
        248_045, 8_678, 198, 24_342, 286, 4_879, 369, 716, 310, 830, 11_553, 13, 5_044, 1_683,
        15_060, 1_472, 279, 3_274, 11, 9_307, 1_328, 30_800, 11, 2_814, 47_675, 25_605, 11, 321,
        60_445, 55_404, 11, 27_224, 11, 321, 30_246, 303, 279, 1_534, 4_087, 13, 248_046, 198,
        248_045, 846, 198, 75_403, 43_466, 0, 248_046, 198, 248_045, 74_455, 198, 248_068, 198,
    ];
    if ids != expected_ids {
        anyhow::bail!("xhigh chat golden differs");
    }
    let options = ChatTemplateOptions {
        enable_thinking: false,
        ..ChatTemplateOptions::default()
    };
    let no_think = render_chat(&[ChatMessage::text(ChatRole::User, "2+2?")], &options)?;
    let no_think_ids = tokenizer.encode(&no_think)?;
    if no_think_ids
        != [
            248_045, 846, 198, 17, 10, 17, 30, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
            248_069, 271,
        ]
    {
        anyhow::bail!("disabled-thinking chat golden differs");
    }
    verify_chat_hashes(&tokenizer)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "format": "ctox.qwen38.tokenizer-verification.v1",
            "status": "passed",
            "tokenizer_sha256": tokenizer.sha256(),
            "chat_template_sha256": tokenizer.chat_template_sha256(),
            "multilingual_text_cases": text_cases.len(),
            "chat_cases": 5,
            "xhigh_rendered_bytes": rendered.len(),
        }))?
    );
    Ok(())
}

fn verify_chat_hashes(tokenizer: &Qwen38Tokenizer) -> anyhow::Result<()> {
    let system_low = [
        ChatMessage::text(ChatRole::System, "Du bist präzise."),
        ChatMessage::text(ChatRole::User, "Antworte kurz."),
    ];
    verify_chat_case(
        tokenizer,
        "system_low",
        &system_low,
        &ChatTemplateOptions {
            reasoning_effort: ReasoningEffort::Low,
            ..ChatTemplateOptions::default()
        },
        "9347c4131e7052dc704819183e09c13aa29e6deb18de3fb3b4f1c13171be2239",
        "5a36693aeea9131e5e3402a9fcd72524deadbd74d3093ce3320057f84721526c",
    )?;

    let history = [
        ChatMessage::text(ChatRole::User, "A"),
        ChatMessage {
            role: ChatRole::Assistant,
            content: Some("B".into()),
            reasoning_content: Some("R".into()),
            tool_calls: Vec::new(),
        },
        ChatMessage::text(ChatRole::User, "C"),
    ];
    verify_chat_case(
        tokenizer,
        "assistant_history",
        &history,
        &ChatTemplateOptions {
            add_generation_prompt: false,
            ..ChatTemplateOptions::default()
        },
        "6eec0286a35e321c057d6b802db50b9e57cd14dba9888aab4d59a2de2414f7dc",
        "a3b883c02d8a47fadbbe629c2f8fc55d28d9c829750bcda5ad10fe00b5e6473d",
    )?;

    let mut arguments = Map::new();
    arguments.insert("city".into(), Value::String("Berlin".into()));
    let tool_flow = [
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
    verify_chat_case(
        tokenizer,
        "tool_flow",
        &tool_flow,
        &ChatTemplateOptions {
            tools: vec![json!({
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            })],
            add_generation_prompt: false,
            ..ChatTemplateOptions::default()
        },
        "74a47458f2dd1baf5574fcb30f542b4d0b8484f56eb5ed98081e1c1cbc9b0df5",
        "170eedafc0bec8db34954a0783d04cc422099412c7e4575a93ae5251df296013",
    )
}

fn verify_chat_case(
    tokenizer: &Qwen38Tokenizer,
    name: &str,
    messages: &[ChatMessage],
    options: &ChatTemplateOptions,
    expected_rendered_sha256: &str,
    expected_ids_sha256: &str,
) -> anyhow::Result<()> {
    let (rendered, ids) = tokenizer.render_and_encode(messages, options)?;
    let rendered_sha256 = format!("{:x}", Sha256::digest(rendered.as_bytes()));
    let mut encoded_ids = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        encoded_ids.extend_from_slice(&id.to_le_bytes());
    }
    let ids_sha256 = format!("{:x}", Sha256::digest(encoded_ids));
    if rendered_sha256 != expected_rendered_sha256 || ids_sha256 != expected_ids_sha256 {
        anyhow::bail!("{name} chat golden differs: rendered={rendered_sha256}, ids={ids_sha256}");
    }
    Ok(())
}
