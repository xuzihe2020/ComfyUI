import json
from pathlib import Path


WORKFLOW = Path("user/default/workflows/vn_foundation_environment_person_flux2_layout.json")


def main():
    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {int(n["id"]): n for n in data.get("nodes", [])}
    links = {int(row[0]): row for row in data.get("links", [])}

    interesting = {
        "Compositor4",
        "CompositorConfig4",
        "Compositor4MasksOutput",
        "ImageMaskSwitch",
        "Mask Invert",
        "ImpactDilateMask",
        "MaskBlur+",
        "MaskToImage",
        "PreviewImage",
        "LoadImage",
    }

    for node in data.get("nodes", []):
        if node.get("type") not in interesting:
            continue
        print(f"NODE {node['id']} {node.get('type')}")
        if node.get("id") in {5, 9, 52, 64}:
            for index, socket in enumerate(node.get("inputs") or []):
                print(f"  SOCKET IN {index} {socket.get('name')} type={socket.get('type')} link={socket.get('link')}")
            for index, socket in enumerate(node.get("outputs") or []):
                print(f"  SOCKET OUT {index} {socket.get('name')} type={socket.get('type')} links={socket.get('links')}")
        for index, socket in enumerate(node.get("inputs") or []):
            link_id = socket.get("link")
            if link_id is None:
                continue
            row = links.get(int(link_id))
            if row:
                src = nodes.get(int(row[1]), {})
                print(
                    f"  IN {index} {socket.get('name')} <- "
                    f"link {link_id} from {row[1]}:{row[2]} {src.get('type')} type={row[5]}"
                )
        for index, socket in enumerate(node.get("outputs") or []):
            out_links = socket.get("links") or []
            if not out_links:
                continue
            targets = []
            for link_id in out_links:
                row = links.get(int(link_id))
                if row:
                    dst = nodes.get(int(row[3]), {})
                    targets.append(f"{link_id}->{row[3]}:{row[4]} {dst.get('type')} type={row[5]}")
            print(f"  OUT {index} {socket.get('name')} -> {', '.join(targets)}")


if __name__ == "__main__":
    main()
