import argparse
import os
import shutil
from pathlib import Path

def _iter_candidate_files(base_dir: Path, pattern: str, *, skip_under: Path | None = None):
    """
    Yield files matching pattern under base_dir (recursive), optionally skipping anything under skip_under.
    """
    for p in base_dir.rglob(pattern):
        if not p.is_file():
            continue
        if skip_under is not None and skip_under in p.parents:
            continue
        yield p


def reorganize_files(input_base_dir, output_base_dir, *, copy_images: bool = True, dry_run: bool = False):
    """
    Reorganize PDF parsing results from structure organized by filename to
    structure organized by file type (json, markdown, images) and then by filename.
    
    Current structure:
    - base_dir/
      - file_name_1/
        - file_name_1.json
        - file_name_1.md
        - images/
          - image1.png
          - image2.png
      - file_name_2/
        ...
    
    New structure:
    - base_dir/
      - json/
        - file_name_1/
          - file_name_1.json
      - markdown/
        - file_name_1/
          - file_name_1.md
      - images/
        - file_name_1/
          - image1.png
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)

    # Create new directories if they don't exist
    json_dir = output_base_dir / "json"
    markdown_dir = output_base_dir / "markdown"
    images_dir = output_base_dir / "images"
    
    if not dry_run:
        for directory in [json_dir, markdown_dir, images_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    # Index markdowns by stem for quick lookup (supports cases where md/json are not siblings)
    md_by_stem: dict[str, Path] = {}
    for md_path in _iter_candidate_files(input_base_dir, "*.md", skip_under=output_base_dir):
        md_by_stem.setdefault(md_path.stem, md_path)

    json_files = list(_iter_candidate_files(input_base_dir, "*_content_list.json", skip_under=output_base_dir))
    md_only_files = list(_iter_candidate_files(input_base_dir, "*.md", skip_under=output_base_dir))

    processed_doc_ids: set[str] = set()
    copied = {"json": 0, "markdown": 0, "images": 0}

    # Primary pass: drive off content_list.json (works for both old and new nested layouts)
    for json_path in json_files:
        name = json_path.name
        if not name.endswith("_content_list.json"):
            continue
        doc_id = name[: -len("_content_list.json")]
        processed_doc_ids.add(doc_id)

        # Copy JSON (strip the _content_list suffix -> <doc_id>.json)
        dest_json = json_dir / f"{doc_id}.json"
        if not dry_run:
            shutil.copy2(json_path, dest_json)
        copied["json"] += 1

        # Copy Markdown: prefer sibling <doc_id>.md, fallback to indexed md anywhere under input
        md_candidate = json_path.parent / f"{doc_id}.md"
        md_path = md_candidate if md_candidate.exists() else md_by_stem.get(doc_id)
        if md_path is not None and md_path.exists():
            dest_md = markdown_dir / f"{doc_id}.md"
            if not dry_run:
                shutil.copy2(md_path, dest_md)
            copied["markdown"] += 1

        # Copy images folder (if present) into images/<doc_id>/
        if copy_images:
            img_folder = json_path.parent / "images"
            if img_folder.exists() and img_folder.is_dir():
                dest_img_dir = images_dir / doc_id
                if not dry_run:
                    dest_img_dir.mkdir(parents=True, exist_ok=True)
                for img in img_folder.iterdir():
                    if img.is_file():
                        if not dry_run:
                            shutil.copy2(img, dest_img_dir / img.name)
                        copied["images"] += 1

    # Secondary pass: handle md-only documents (no content_list.json found)
    for md_path in md_only_files:
        doc_id = md_path.stem
        if doc_id in processed_doc_ids:
            continue
        processed_doc_ids.add(doc_id)

        dest_md = markdown_dir / f"{doc_id}.md"
        if not dry_run:
            shutil.copy2(md_path, dest_md)
        copied["markdown"] += 1

        # If there is a sibling content_list.json, copy it too
        json_candidate = md_path.parent / f"{doc_id}_content_list.json"
        if json_candidate.exists():
            dest_json = json_dir / f"{doc_id}.json"
            if not dry_run:
                shutil.copy2(json_candidate, dest_json)
            copied["json"] += 1

        if copy_images:
            img_folder = md_path.parent / "images"
            if img_folder.exists() and img_folder.is_dir():
                dest_img_dir = images_dir / doc_id
                if not dry_run:
                    dest_img_dir.mkdir(parents=True, exist_ok=True)
                for img in img_folder.iterdir():
                    if img.is_file():
                        if not dry_run:
                            shutil.copy2(img, dest_img_dir / img.name)
                        copied["images"] += 1

    if dry_run:
        print(f"[dry-run] discovered {len(processed_doc_ids)} documents")
        print(f"[dry-run] would copy: json={copied['json']}, markdown={copied['markdown']}, images={copied['images']}")
    else:
        print(f"Copied: json={copied['json']}, markdown={copied['markdown']}, images={copied['images']}")

def clean_paths_in_files(
    directory,
    base_paths_to_remove=(
        "/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/",
        "/data/home/xiong/data/Fund/pdf_parse/Fund_Document/output/",
    ),
    file_patterns=("**/*.md", "**/*.json"),
):
    """
    Process files in a directory and replace paths within their content.
    
    Args:
        directory (str or Path): The directory containing files to process
        base_paths_to_remove (str | list | tuple): Base path(s) to remove from file contents
        file_patterns (list): List of glob patterns for files to process
        
    Returns:
        int: Number of files processed
    """
    # Ensure directory is a Path object
    directory = Path(directory)
    
    # Find all matching files
    all_files = []
    for pattern in file_patterns:
        all_files.extend(list(directory.glob(pattern)))
    
    files_processed = 0
    
    # Normalize base paths
    if isinstance(base_paths_to_remove, (str, Path)):
        base_paths = [str(base_paths_to_remove)]
    else:
        base_paths = [str(p) for p in base_paths_to_remove]

    # Process each file
    for file_path in all_files:
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the paths in the content
            new_content = content
            for base_path in base_paths:
                new_content = new_content.replace(base_path, "")
            
            # Write the modified content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            files_processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return files_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process MinerU PDF parsing results (reorganize + path cleanup).")
    parser.add_argument(
        "--input_base_dir",
        type=str,
        default="/data/home/xiong/data/Fund/pdf_parse/Fund_Document/output/missing_program_json_md",
        help="Input directory containing parsed outputs (supports nested layouts).",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/data/home/xiong/data/Fund/pdf_parse/Fund_Document/output_organized/missing_program_json_md",
        help="Output directory to write the reorganized structure (json/markdown/images).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Discover and report counts without copying files.")
    parser.add_argument("--no_images", action="store_true", help="Skip copying images.")
    parser.add_argument(
        "--base_path_to_remove",
        action="append",
        default=[],
        help="Base path to remove from file contents (can be repeated).",
    )
    args = parser.parse_args()

    input_base_dir = Path(args.input_base_dir)
    output_base_dir = Path(args.output_base_dir)

    if not input_base_dir.exists():
        raise FileNotFoundError(f"input_base_dir does not exist: {input_base_dir}")

    if not args.dry_run:
        output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reorganizing files in {input_base_dir} -> {output_base_dir} ...")
    reorganize_files(
        input_base_dir,
        output_base_dir,
        copy_images=not args.no_images,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[dry-run] skipping in-place content path cleanup")
    else:
        print("Clearing paths in files...")
        extra_paths = tuple(args.base_path_to_remove) if args.base_path_to_remove else tuple()
        base_paths = (
            "/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/",
            "/data/home/xiong/data/Fund/pdf_parse/Fund_Document/output/",
            "_json_md",
        ) + extra_paths
        files_processed = clean_paths_in_files(output_base_dir, base_paths_to_remove=base_paths)
        print(f"Processed {files_processed} files to clean paths.")
    