use std::time::{Duration, Instant};

use bytes::Bytes;
use omp_core::Str;
use omp_proto::thread::v1::{Item, item};
use omp_tool::Rev;
use omp_tui::components::{ToolCard, ToolState};
use omp_tui::components::{DiffKind, DiffView};
use omp_tui::components::TextLeaf;
use omp_tui::components::Col;
use omp_tui::Ui;

/// The retained state for a single tool call.
#[derive(Debug)]
pub(crate) struct ToolFold {
    pub(crate) call_id: Str,
    pub(crate) name: Str,
    pub(crate) rev: Rev,
    pub(crate) args_slop: String,
    pub(crate) parsed_args: omp_slopjson::Value,
    pub(crate) updates: Vec<serde_json::Value>,
    pub(crate) item: Option<Item>,
    pub(crate) start_time: Instant,
    pub(crate) elapsed: Duration,
    pub(crate) state: ToolState,
}

impl ToolFold {
    pub(crate) fn new(call_id: Str, name: Str, rev: Rev) -> Self {
        Self {
            call_id,
            name,
            rev,
            args_slop: String::new(),
            parsed_args: omp_slopjson::Value::Object(omp_slopjson::Object::new()),
            updates: Vec::new(),
            item: None,
            start_time: Instant::now(),
            elapsed: Duration::default(),
            state: ToolState::Streaming,
        }
    }

    pub(crate) fn push_args(&mut self, fragment: &str) {
        self.args_slop.push_str(fragment);
        self.parsed_args = omp_slopjson::parse_streaming(&self.args_slop);
    }

    pub(crate) fn push_update(&mut self, json: Bytes) {
        if let Ok(val) = serde_json::from_slice(&json) {
            self.updates.push(val);
        }
    }
}

pub(crate) type RendererFn = fn(&mut Ui, &ToolFold) -> bool;

pub(crate) struct RendererRegistry {
    renderers: &'static [(&'static str, &'static str, RendererFn)],
}

impl RendererRegistry {
    pub(crate) const fn new() -> Self {
        Self {
            renderers: &[
                ("read", "", render_read),
                ("edit", "hl", render_edit),
                ("shell", "", render_shell),
                ("grep", "", render_grep),
                ("glob", "", render_glob),
            ],
        }
    }

    pub(crate) fn update(&self, ui: &mut Ui, fold: &ToolFold) -> bool {
        for (name, family, f) in self.renderers {
            if fold.name.as_str() == *name && fold.rev.family.as_str() == *family {
                return f(ui, fold);
            }
        }
        render_generic(ui, fold)
    }
}

fn proto_to_json(v: &omp_proto::inference::v1::Value) -> Option<serde_json::Value> {
    match v.kind.as_ref()? {
        omp_proto::inference::v1::value::Kind::Null(_) => Some(serde_json::Value::Null),
        omp_proto::inference::v1::value::Kind::Int(n) => Some(serde_json::Value::from(*n)),
        omp_proto::inference::v1::value::Kind::Uint(n) => Some(serde_json::Value::from(*n)),
        omp_proto::inference::v1::value::Kind::Double(n) => serde_json::Number::from_f64(*n).map(serde_json::Value::Number),
        omp_proto::inference::v1::value::Kind::Bool(b) => Some(serde_json::Value::Bool(*b)),
        omp_proto::inference::v1::value::Kind::String(s) => Some(serde_json::Value::String(s.clone())),
        omp_proto::inference::v1::value::Kind::List(list) => {
            let mut out = Vec::with_capacity(list.values.len());
            for item in &list.values {
                out.push(proto_to_json(item)?);
            }
            Some(serde_json::Value::Array(out))
        }
        omp_proto::inference::v1::value::Kind::Map(m) => {
            let mut map = serde_json::Map::with_capacity(m.fields.len());
            for (k, v) in &m.fields {
                map.insert(k.clone(), proto_to_json(v)?);
            }
            Some(serde_json::Value::Object(map))
        }
    }
}

fn get_payload<T: serde::de::DeserializeOwned>(item: &Option<Item>) -> Option<T> {
    let i = item.as_ref()?;
    let item::Kind::ToolResult(res) = &i.kind.as_ref()? else { return None; };
    let details = res.details.as_ref()?;
    let json_val = proto_to_json(details)?;
    serde_json::from_value(json_val).ok()
}

pub(crate) fn render_read(ui: &mut Ui, fold: &ToolFold) -> bool {
    ui.update_component::<ToolCard>(&fold.call_id, |c| {
        c.set_name(fold.name.clone());
        c.set_state(fold.state);
        
        if let Some(path) = fold.parsed_args.get("path").and_then(|v| v.as_str()) {
            c.set_intent(format!("read {}", path));
        }
        
        if fold.state == ToolState::Success {
            if let Some(payload) = get_payload::<omp_tools::read::Payload>(&fold.item) {
                let mut col = Col::new();
                match payload.content {
                    omp_tools::read::Content::Text { slices } => {
                        for slice in slices {
                            col = col.child(TextLeaf::new().text(slice.text.as_str()));
                        }
                    }
                    omp_tools::read::Content::Summary { segments: _ } => {
                        col = col.child(TextLeaf::new().text("Structural Summary"));
                    }
                    omp_tools::read::Content::Blob { fallback, .. } => {
                        col = col.child(TextLeaf::new().text(fallback.as_str()));
                    }
                }
                c.replace_body(col);
            }
        }
        true
    })
}

pub(crate) fn render_edit(ui: &mut Ui, fold: &ToolFold) -> bool {
    ui.update_component::<ToolCard>(&fold.call_id, |c| {
        c.set_name(fold.name.clone());
        c.set_state(fold.state);
        
        if let Some(path) = fold.parsed_args.get("path").and_then(|v| v.as_str()) {
            c.set_intent(format!("edit {}", path));
        }
        
        if fold.state == ToolState::Streaming {
            if let Some(last_update) = fold.updates.last() {
                if let Some(preview) = last_update.get("preview").and_then(|v| v.as_str()) {
                    let mut diff = DiffView::new();
                    for line in preview.lines() {
                        diff.push(DiffKind::Context, line);
                    }
                    c.replace_body(diff);
                }
            }
        } else if fold.state == ToolState::Success {
            if let Some(payload) = get_payload::<omp_tools::edit::Payload>(&fold.item) {
                let mut diff = DiffView::new();
                for line in payload.diff.lines() {
                    if line.starts_with('+') && !line.starts_with("+++") {
                        diff.push(DiffKind::Add, line);
                    } else if line.starts_with('-') && !line.starts_with("---") {
                        diff.push(DiffKind::Remove, line);
                    } else if line.starts_with('@') || line.starts_with("---") || line.starts_with("+++") {
                        diff.push(DiffKind::Header, line);
                    } else {
                        diff.push(DiffKind::Context, line);
                    }
                }
                c.replace_body(diff);
            }
        }
        true
    })
}

pub(crate) fn render_shell(ui: &mut Ui, fold: &ToolFold) -> bool {
    ui.update_component::<ToolCard>(&fold.call_id, |c| {
        c.set_name(fold.name.clone());
        c.set_state(fold.state);
        
        if let Some(command) = fold.parsed_args.get("command").and_then(|v| v.as_str()) {
            c.set_intent(command.split('\n').next().unwrap_or(command));
        }

        let mut tail_text = String::new();
        // Capped live tail
        let recent_updates = fold.updates.iter().rev().take(10).collect::<Vec<_>>();
        for u in recent_updates.into_iter().rev() {
            if let Some(text) = u.get("text").and_then(|v| v.as_str()) {
                tail_text.push_str(text);
            }
        }

        if fold.state != ToolState::Streaming {
            if let Some(payload) = get_payload::<omp_tools::shell::Payload>(&fold.item) {
                match payload.status.outcome {
                    omp_tools::shell::ExecOutcome::Exited => {
                        if let Some(code) = payload.status.exit_code {
                            if code != 0 {
                                c.set_badge(format!("exit {}", code));
                            }
                        }
                    }
                    omp_tools::shell::ExecOutcome::Cancelled => {
                        c.set_badge("interrupted");
                    }
                    omp_tools::shell::ExecOutcome::Timeout => {
                        c.set_badge("timeout");
                    }
                    _ => {}
                }
            }
        }

        if !tail_text.is_empty() {
            let lines: Vec<&str> = tail_text.lines().rev().take(5).collect();
            let tail_preview = lines.into_iter().rev().collect::<Vec<_>>().join("\n");
            let col = Col::new().child(TextLeaf::new().text(tail_preview));
            c.replace_body(col);
        } else {
            c.replace_body(Col::new());
        }
        true
    })
}

pub(crate) fn render_grep(ui: &mut Ui, fold: &ToolFold) -> bool {
    ui.update_component::<ToolCard>(&fold.call_id, |c| {
        c.set_name(fold.name.clone());
        c.set_state(fold.state);
        
        if let Some(query) = fold.parsed_args.get("query").and_then(|v| v.as_str()) {
            c.set_intent(format!("grep {}", query));
        }

        if fold.state == ToolState::Success {
            if let Some(payload) = get_payload::<omp_tools::grep::Payload>(&fold.item) {
                let mut col = Col::new();
                for m in payload.matches {
                    col = col.child(TextLeaf::new().text(format!("{}:{}", m.path.as_str(), m.line)));
                }
                if payload.truncated {
                    col = col.child(TextLeaf::new().text("..."));
                }
                c.replace_body(col);
            }
        }
        true
    })
}

pub(crate) fn render_glob(ui: &mut Ui, fold: &ToolFold) -> bool {
    ui.update_component::<ToolCard>(&fold.call_id, |c| {
        c.set_name(fold.name.clone());
        c.set_state(fold.state);
        
        if let Some(path) = fold.parsed_args.get("path").and_then(|v| v.as_str()) {
            c.set_intent(format!("glob {}", path));
        }

        if fold.state == ToolState::Success {
            if let Some(payload) = get_payload::<omp_tools::glob::Payload>(&fold.item) {
                let mut col = Col::new();
                for p in payload.paths {
                    col = col.child(TextLeaf::new().text(p.as_str()));
                }
                if payload.truncated {
                    col = col.child(TextLeaf::new().text("..."));
                }
                c.replace_body(col);
            }
        }
        true
    })
}

pub(crate) fn render_generic(ui: &mut Ui, fold: &ToolFold) -> bool {
    ui.update_component::<ToolCard>(&fold.call_id, |c| {
        c.set_name(fold.name.clone());
        c.set_state(fold.state);
        
        if let Some(intent) = fold.parsed_args.get("path").or_else(|| fold.parsed_args.get("command")).or_else(|| fold.parsed_args.get("query")).and_then(|v| v.as_str()) {
            c.set_intent(intent);
        }
        
        if fold.state != ToolState::Streaming {
            c.set_badge(format!("{}ms", fold.elapsed.as_millis()));
        }
        
        if fold.state == ToolState::Success {
            if let Some(payload) = get_payload::<serde_json::Value>(&fold.item) {
                let text = serde_json::to_string_pretty(&payload).unwrap_or_default();
                let lines: Vec<&str> = text.lines().take(10).collect();
                let mut preview = lines.join("\n");
                if text.lines().count() > 10 {
                    preview.push_str("\n...");
                }
                c.replace_body(Col::new().child(TextLeaf::new().text(preview)));
            }
        } else if fold.state == ToolState::Failure {
            if let Some(i) = fold.item.as_ref() {
                if let Some(item::Kind::ToolResult(res)) = &i.kind {
                    if let Some(details) = &res.details {
                        if let Some(json) = proto_to_json(details) {
                            let text = serde_json::to_string_pretty(&json).unwrap_or_default();
                            c.replace_body(Col::new().child(TextLeaf::new().text(text)));
                        }
                    }
                }
            }
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use omp_tui::{UiContext, test_support::frame_row_text};

    #[test]
    fn test_behavior_generic_card() {
        let mut ui = Ui::from_root(
            ToolCard::new().with(omp_tui::Prop::Id, "t1"),
            40,
            UiContext::default(),
        );
        
        let mut fold = ToolFold::new("t1".into(), "unknown".into(), Rev { family: "".into(), n: 1 });
        fold.push_args("{\"path\": \"foo.txt\"}");
        
        let reg = RendererRegistry::new();
        assert!(reg.update(&mut ui, &fold));
        let header = frame_row_text(ui.frame(), 0);
        assert!(header.contains("unknown"));
        assert!(header.contains("foo.txt"));
    }
}
