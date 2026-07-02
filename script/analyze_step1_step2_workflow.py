import json
from pathlib import Path


PATH = Path("user/default/workflows/dev/vn_step1_step2_background_plate_flux2.json")


def main() -> int:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}

    def source_of(node, input_name):
        for index, inp in enumerate(node.get("inputs", [])):
            if inp.get("name") == input_name:
                link_id = inp.get("link")
                if link_id is None:
                    return None
                link = links[link_id]
                src = nodes[link[1]]
                return src, link[2], link_id
        return None

    for node in data["nodes"]:
        title = node.get("title", "")
        if "stage2 background plate" in title or node["type"] in {"SetLatentNoiseMask", "SamplerCustomAdvanced", "SplitSigmasDenoise"}:
            print(f"id={node['id']} type={node['type']} title={title!r}")
            for inp in node.get("inputs", []):
                link_id = inp.get("link")
                src_desc = ""
                if link_id is not None:
                    link = links[link_id]
                    src = nodes[link[1]]
                    src_desc = f" <- id={src['id']} type={src['type']} title={src.get('title', '')!r} out={link[2]}"
                print(f"  in {inp.get('name')} type={inp.get('type')} link={link_id}{src_desc}")
            print(f"  widgets={node.get('widgets_values')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
