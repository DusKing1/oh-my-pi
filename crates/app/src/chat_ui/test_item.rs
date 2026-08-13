use omp_proto::thread::v1::{Item, item::Kind, ToolResult};
pub fn extract_payload(item: &Item) -> Option<&omp_proto::inference::v1::Value> {
    if let Some(Kind::ToolResult(tr)) = &item.kind {
        tr.details.as_ref()
    } else {
        None
    }
}
