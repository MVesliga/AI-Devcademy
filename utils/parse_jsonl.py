import json

def parse_jsonl_line(line, text_field, doc_id_field, author_field):
    try:
        record = json.loads(line.strip())
        return (
            record.get(doc_id_field),
            record.get(author_field),
            record.get(text_field, "").strip()
        )
    except json.JSONDecodeError:
        return None, None, None