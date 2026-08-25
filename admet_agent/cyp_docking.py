"""CYP-focused AutoDock Vina docking and PyMOL structural explanation.

The ADMET models decide *which* CYP isoform is worth inspecting. This module
then treats docking as a separate, hypothesis-generating workflow: it obtains
or accepts a receptor structure, prepares a ligand, runs Vina, and creates
reproducible PyMOL figures and a machine-readable manifest.

The implementation deliberately keeps the heme cofactor. Removing HEM is a
common generic-receptor-preparation mistake that makes CYP poses meaningless.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class CYPDockingError(RuntimeError):
    """Raised when a CYP docking stage cannot produce a valid artifact."""


@dataclass(frozen=True)
class CYPReceptorSpec:
    isoform: str
    pdb_id: str
    chain: str
    description: str


# Ligand-bound experimental structures are preferred where a practical default
# exists. Users can always override these with --pdb-id or --receptor-pdb.
CYP_RECEPTORS: dict[str, CYPReceptorSpec] = {
    "CYP1A2": CYPReceptorSpec(
        "CYP1A2", "2HI4", "A", "Human CYP1A2 experimental structure"
    ),
    "CYP2C9": CYPReceptorSpec(
        "CYP2C9", "1OG5", "A", "Human CYP2C9 experimental structure"
    ),
    "CYP2D6": CYPReceptorSpec(
        "CYP2D6", "3TBG", "A", "Human CYP2D6 ligand-bound structure"
    ),
    "CYP3A4": CYPReceptorSpec(
        "CYP3A4", "1W0F", "A", "Human CYP3A4 progesterone-bound structure"
    ),
}


@dataclass(frozen=True)
class CYPDockingConfig:
    isoform: str
    output_dir: Path
    compound_name: str = "ligand"
    smiles: str | None = None
    ligand_sdf: Path | None = None
    pubchem_cid: int | None = None
    pdb_id: str | None = None
    receptor_pdb: Path | None = None
    receptor_pdbqt: Path | None = None
    chain: str | None = None
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] = (26.0, 26.0, 26.0)
    receptor_prep: str = "auto"
    cpu: int = 4
    exhaustiveness: int = 32
    num_modes: int = 9
    energy_range: float = 4.0
    seed: int = 20260824
    render: bool = True
    pymol_bin: str | None = None
    image_size: tuple[int, int] = (1800, 1400)


def normalize_isoform(value: str) -> str:
    normalized = value.upper().replace("-", "").replace("_", "")
    if not normalized.startswith("CYP"):
        normalized = f"CYP{normalized}"
    if normalized not in CYP_RECEPTORS:
        supported = ", ".join(CYP_RECEPTORS)
        raise ValueError(f"Unsupported CYP isoform {value!r}; choose one of {supported}")
    return normalized


def validate_config(config: CYPDockingConfig) -> CYPDockingConfig:
    isoform = normalize_isoform(config.isoform)
    if any(character.isspace() for character in str(config.output_dir)):
        raise ValueError("output_dir must not contain whitespace (PyMOL limitation)")
    ligand_inputs = [
        config.smiles is not None,
        config.ligand_sdf is not None,
        config.pubchem_cid is not None,
    ]
    if sum(ligand_inputs) != 1:
        raise ValueError(
            "Provide exactly one ligand source: smiles, ligand_sdf, or pubchem_cid"
        )
    if config.ligand_sdf is not None and not config.ligand_sdf.is_file():
        raise FileNotFoundError(config.ligand_sdf)
    if config.receptor_pdb is not None and not config.receptor_pdb.is_file():
        raise FileNotFoundError(config.receptor_pdb)
    if config.receptor_pdbqt is not None and not config.receptor_pdbqt.is_file():
        raise FileNotFoundError(config.receptor_pdbqt)
    if config.receptor_prep not in {"auto", "meeko", "openbabel"}:
        raise ValueError("receptor_prep must be 'auto', 'meeko', or 'openbabel'")
    if config.cpu < 1 or config.exhaustiveness < 1 or config.num_modes < 1:
        raise ValueError("cpu, exhaustiveness, and num_modes must be positive")
    if config.energy_range <= 0:
        raise ValueError("energy_range must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in config.box_size):
        raise ValueError("All box dimensions must be finite and positive")
    return CYPDockingConfig(**{**asdict(config), "isoform": isoform})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ADMET-EvO-CYP-structural-explanation/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())
    except Exception as exc:
        raise CYPDockingError(f"Failed to download {url}: {exc}") from exc
    if destination.stat().st_size == 0:
        raise CYPDockingError(f"Downloaded an empty file from {url}")


def download_pdb(pdb_id: str, destination: Path) -> None:
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")
    _download(f"https://files.rcsb.org/download/{pdb_id}.pdb", destination)
    lines = destination.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in lines):
        raise CYPDockingError(f"RCSB response for {pdb_id} is not a coordinate PDB")


def download_pubchem_3d_sdf(cid: int, destination: Path) -> None:
    if cid <= 0:
        raise ValueError("PubChem CID must be positive")
    _download(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
        "?record_type=3d",
        destination,
    )


def prepare_ligand_sdf(
    destination: Path,
    *,
    smiles: str | None,
    ligand_sdf: Path | None,
    pubchem_cid: int | None,
    compound_name: str,
    seed: int,
) -> None:
    if ligand_sdf is not None:
        shutil.copy2(ligand_sdf, destination)
        return
    if pubchem_cid is not None:
        download_pubchem_3d_sdf(pubchem_cid, destination)
        return

    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles or "")
    if molecule is None:
        raise CYPDockingError("RDKit could not parse the provided SMILES")
    molecule = Chem.AddHs(molecule)
    embed = AllChem.ETKDGv3()
    embed.randomSeed = int(seed) & 0x7FFFFFFF
    if AllChem.EmbedMolecule(molecule, embed) != 0:
        raise CYPDockingError("RDKit could not generate a 3D ligand conformer")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000)
    else:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=1000)
    molecule.SetProp("_Name", compound_name)
    writer = Chem.SDWriter(str(destination))
    writer.write(molecule)
    writer.close()


def clean_cyp_receptor(
    source: Path,
    destination: Path,
    *,
    chain: str,
) -> dict[str, int | str]:
    """Keep one protein chain and its heme, while removing ligands and water."""

    kept: list[str] = []
    protein_atoms = 0
    heme_atoms = 0
    removed_hetero_atoms = 0
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[:6]
        if record == "ATOM  ":
            if len(line) > 21 and line[21].strip() == chain:
                kept.append(line)
                protein_atoms += 1
            continue
        if record == "HETATM":
            residue = line[17:20].strip().upper()
            line_chain = line[21].strip() if len(line) > 21 else ""
            if residue == "HEM" and line_chain in {chain, ""}:
                kept.append(line)
                heme_atoms += 1
            else:
                removed_hetero_atoms += 1
    if protein_atoms == 0:
        raise CYPDockingError(f"No protein atoms found for chain {chain!r}")
    if heme_atoms == 0:
        raise CYPDockingError(
            f"No HEM cofactor found for chain {chain!r}; refusing CYP docking"
        )
    kept.extend(["TER", "END"])
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return {
        "chain": chain,
        "protein_atoms": protein_atoms,
        "heme_atoms": heme_atoms,
        "removed_hetero_atoms": removed_hetero_atoms,
    }


def heme_iron_coordinate(receptor_pdb: Path) -> tuple[float, float, float]:
    for line in receptor_pdb.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith("HETATM"):
            continue
        if line[17:20].strip().upper() != "HEM":
            continue
        if line[12:16].strip().upper() != "FE":
            continue
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    raise CYPDockingError("The cleaned receptor has HEM but no FE atom")


def _resolve_executable(name: str, explicit: str | None = None) -> str:
    candidate = explicit or shutil.which(name)
    if candidate is None:
        raise CYPDockingError(
            f"Required executable {name!r} was not found. "
            "Create the environment from environment-docking.yml."
        )
    return candidate


def _run_command(
    command: Sequence[str],
    *,
    log_handle,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    log_handle.write(f"$ {' '.join(command)}\n")
    log_handle.flush()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_handle.write(completed.stdout)
    if completed.stdout and not completed.stdout.endswith("\n"):
        log_handle.write("\n")
    log_handle.flush()
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise CYPDockingError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{tail}"
        )
    return completed


def _prepare_receptor_pdbqt(
    cleaned_pdb: Path,
    output_pdbqt: Path,
    *,
    method: str,
    log_handle,
    cwd: Path,
) -> str:
    if method == "meeko":
        executable = _resolve_executable("mk_prepare_receptor.py")
        prefix = output_pdbqt.with_suffix("")
        _run_command(
            [executable, "-i", str(cleaned_pdb), "-o", str(prefix), "-p"],
            log_handle=log_handle,
            cwd=cwd,
        )
        generated = prefix.with_suffix(".pdbqt")
        if generated != output_pdbqt and generated.exists():
            shutil.move(generated, output_pdbqt)
    else:
        executable = _resolve_executable("obabel")
        _run_command(
            [
                executable,
                str(cleaned_pdb),
                "-O",
                str(output_pdbqt),
                "-xr",
                "-p",
                "7.4",
            ],
            log_handle=log_handle,
            cwd=cwd,
        )
    if not output_pdbqt.is_file() or output_pdbqt.stat().st_size == 0:
        raise CYPDockingError("Receptor preparation did not create receptor.pdbqt")
    return executable


def _prepare_ligand_pdbqt(
    ligand_sdf: Path,
    output_pdbqt: Path,
    *,
    log_handle,
    cwd: Path,
) -> str:
    executable = _resolve_executable("mk_prepare_ligand.py")
    _run_command(
        [executable, "-i", str(ligand_sdf), "-o", str(output_pdbqt)],
        log_handle=log_handle,
        cwd=cwd,
    )
    if not output_pdbqt.is_file() or output_pdbqt.stat().st_size == 0:
        raise CYPDockingError("Ligand preparation did not create ligand.pdbqt")
    return executable


def parse_vina_scores(pdbqt_path: Path) -> list[dict[str, float | int]]:
    scores: list[dict[str, float | int]] = []
    for line in pdbqt_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith("REMARK VINA RESULT:"):
            continue
        values = line.split(":", 1)[1].split()
        if len(values) < 3:
            continue
        scores.append(
            {
                "mode": len(scores) + 1,
                "affinity_kcal_mol": float(values[0]),
                "rmsd_lb_angstrom": float(values[1]),
                "rmsd_ub_angstrom": float(values[2]),
            }
        )
    if not scores:
        raise CYPDockingError("No REMARK VINA RESULT scores were found in docking output")
    return scores


def _export_vina_poses(
    docking_pdbqt: Path,
    poses_sdf: Path,
    best_pose_sdf: Path,
    *,
    log_handle,
    cwd: Path,
) -> str:
    executable = _resolve_executable("mk_export.py")
    _run_command(
        [executable, str(docking_pdbqt), "-s", str(poses_sdf)],
        log_handle=log_handle,
        cwd=cwd,
    )
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)
    best = next((molecule for molecule in supplier if molecule is not None), None)
    if best is None:
        raise CYPDockingError("Meeko export produced no readable docking pose")
    writer = Chem.SDWriter(str(best_pose_sdf))
    writer.write(best)
    writer.close()
    return executable


def closest_ligand_atom_to_heme(
    receptor_pdb: Path, ligand_sdf: Path
) -> dict[str, float | int | str]:
    from rdkit import Chem

    molecule = next(
        (mol for mol in Chem.SDMolSupplier(str(ligand_sdf), removeHs=False) if mol),
        None,
    )
    if molecule is None or molecule.GetNumConformers() == 0:
        raise CYPDockingError("Best pose SDF has no readable 3D conformer")
    iron = heme_iron_coordinate(receptor_pdb)
    conformer = molecule.GetConformer()
    best: tuple[float, int] | None = None
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        distance = math.dist(iron, (point.x, point.y, point.z))
        if best is None or distance < best[0]:
            best = (distance, atom.GetIdx())
    if best is None:
        raise CYPDockingError("Best pose contains no heavy atoms")
    atom = molecule.GetAtomWithIdx(best[1])
    return {
        "atom_index_zero_based": best[1],
        "atom_rank_for_pymol": best[1],
        "element": atom.GetSymbol(),
        "fe_distance_angstrom": best[0],
    }


def write_scores_csv(path: Path, rows: Iterable[dict[str, float | int]]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "mode",
                "affinity_kcal_mol",
                "rmsd_lb_angstrom",
                "rmsd_ub_angstrom",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_complex_pdb(receptor_pdb: Path, ligand_sdf: Path, destination: Path) -> None:
    from rdkit import Chem

    molecule = next(
        (mol for mol in Chem.SDMolSupplier(str(ligand_sdf), removeHs=False) if mol),
        None,
    )
    if molecule is None:
        raise CYPDockingError("Cannot build complex PDB from unreadable ligand SDF")
    receptor_lines = [
        line
        for line in receptor_pdb.read_text(encoding="utf-8").splitlines()
        if line not in {"END", "TER"}
    ]
    ligand_lines = []
    for line in Chem.MolToPDBBlock(molecule).splitlines():
        if line.startswith("HETATM"):
            line = f"{line[:17]}LIG{line[20:]}"
            ligand_lines.append(line)
    destination.write_text(
        "\n".join([*receptor_lines, "TER", *ligand_lines, "END"]) + "\n",
        encoding="utf-8",
    )


def _pml_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace('"', '\\"')


def build_cyp_pymol_script(
    *,
    receptor_pdb: Path,
    best_pose_sdf: Path,
    output_dir: Path,
    closest_atom_rank: int,
    width: int,
    height: int,
) -> str:
    """Build the protein-ligand PML recipe from pymol-figure-rendering."""

    receptor = _pml_path(receptor_pdb)
    ligand = _pml_path(best_pose_sdf)
    overall = _pml_path(output_dir / "figure_a_overall_pose.png")
    site = _pml_path(output_dir / "figure_b_binding_site.png")
    geometry = _pml_path(output_dir / "figure_c_heme_geometry.png")
    session = _pml_path(output_dir / "pymol_docking_session.pse")
    return f"""reinitialize
load "{receptor}", source_receptor
load "{ligand}", ligand
create receptor, source_receptor and polymer.protein
create heme, source_receptor and resn HEM
create ligand_surface, ligand
delete source_receptor
hide everything
bg_color white
set ray_opaque_background, on
set orthoscopic, on
set depth_cue, 0
set specular, 0
set ambient, 0.42
set ray_shadow, off
set ray_trace_mode, 1
set ray_trace_color, black
set antialias, 2
set valence, off
set cartoon_highlight_color, wheat
set_color pocket_teal, [0.20, 0.62, 0.68]
set_color ligand_carbon, [0.88, 0.30, 0.24]
set_color heme_carbon, [0.72, 0.28, 0.34]
select pocket, byres (receptor within 5.0 of ligand)
select contact_residues, byres (receptor within 4.0 of ligand)
show cartoon, receptor
show surface, receptor
show sticks, ligand or heme
color palecyan, receptor
color heme_carbon, heme and elem C
color ligand_carbon, ligand and elem C
color blue, (ligand or heme) and elem N
color red, (ligand or heme) and elem O
color tv_orange, heme and name FE
set stick_radius, 0.18, ligand
set stick_radius, 0.13, heme
set transparency, 0.18, receptor
hide surface, pocket
orient receptor
zoom receptor, 5
turn y, 18
turn x, -8
ray {width}, {height}
png {overall}, dpi=300
hide everything
show cartoon, receptor
show sticks, ligand or heme or contact_residues
show surface, ligand_surface
color palecyan, receptor
color pocket_teal, contact_residues and elem C
color red, contact_residues and elem O
color blue, contact_residues and elem N
color ligand_carbon, ligand and elem C
color red, ligand and elem O
color blue, ligand and elem N
color heme_carbon, heme and elem C
color tv_orange, heme and name FE
color red, ligand_surface
set transparency, 0.72, ligand_surface
set stick_radius, 0.16, contact_residues
set stick_radius, 0.22, ligand
distance polar_contacts, ligand, contact_residues, 3.6, mode=2
set dash_color, yellow, polar_contacts
set dash_radius, 0.045, polar_contacts
hide labels, polar_contacts
label contact_residues and name CA, resn + resi
set label_size, 14
set label_color, black
set label_outline_color, white
orient ligand or heme
zoom ligand or heme or pocket, 5.5
ray {width}, {height}
png {site}, dpi=300
hide everything
show sticks, ligand or heme or pocket
color pocket_teal, pocket and elem C
color red, pocket and elem O
color blue, pocket and elem N
color ligand_carbon, ligand and elem C
color red, ligand and elem O
color blue, ligand and elem N
color heme_carbon, heme and elem C
color tv_orange, heme and name FE
distance fe_to_ligand, heme and name FE, ligand and rank {closest_atom_rank}
set dash_color, yellow, fe_to_ligand
set dash_radius, 0.075, fe_to_ligand
set dash_gap, 0.25, fe_to_ligand
set label_size, 22
set label_color, black
set label_outline_color, white
orient ligand or heme
zoom ligand or heme or pocket, 3.2
ray {width}, {height}
png {geometry}, dpi=300
save {session}
quit
"""


def compose_figure(output_dir: Path, title: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    inputs = [
        output_dir / "figure_a_overall_pose.png",
        output_dir / "figure_b_binding_site.png",
        output_dir / "figure_c_heme_geometry.png",
    ]
    images = [Image.open(path).convert("RGB") for path in inputs]
    panel_width = 900
    panel_height = 700
    header = 100
    canvas = Image.new("RGB", (panel_width * 3, panel_height + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)
    label_font = ImageFont.load_default(size=22)
    draw.text((36, 24), title, fill="black", font=font)
    labels = ("A  Overall pose", "B  Binding site", "C  Heme geometry")
    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail((panel_width - 20, panel_height - 20))
        x = index * panel_width + (panel_width - image.width) // 2
        y = header + (panel_height - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(
            (index * panel_width + 24, header + 18),
            label,
            fill="black",
            font=label_font,
        )
    output = output_dir / "cyp_docking_composite.png"
    canvas.save(output, dpi=(300, 300))
    return output


def _artifact_manifest(output_dir: Path) -> list[dict[str, str | int]]:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        artifacts.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return artifacts


def run_cyp_docking(config: CYPDockingConfig) -> dict[str, object]:
    config = validate_config(config)
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    spec = CYP_RECEPTORS[config.isoform]
    pdb_id = (config.pdb_id or spec.pdb_id).upper()
    chain = config.chain or spec.chain

    receptor_original = output_dir / "receptor_original.pdb"
    receptor_clean = output_dir / "receptor_clean_with_heme.pdb"
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    ligand_input = output_dir / "ligand_input.sdf"
    ligand_pdbqt = output_dir / "ligand.pdbqt"
    docking_output = output_dir / "docked_poses.pdbqt"
    poses_sdf = output_dir / "docked_poses.sdf"
    best_pose_sdf = output_dir / "best_pose.sdf"
    scores_csv = output_dir / "docking_scores.csv"
    complex_pdb = output_dir / "complex_best_pose.pdb"
    pml_path = output_dir / "render_cyp_docking.pml"
    log_path = output_dir / "pipeline.log"

    warnings = [
        "Docking poses and Vina scores are hypotheses, not experimental evidence.",
        "A single rigid CYP structure does not represent conformational dynamics.",
        "Vina affinity scores must not be interpreted as measured binding free energies.",
    ]
    tools: dict[str, str] = {}

    try:
        if config.receptor_pdb is not None:
            shutil.copy2(config.receptor_pdb, receptor_original)
        else:
            download_pdb(pdb_id, receptor_original)
        cleaning = clean_cyp_receptor(
            receptor_original, receptor_clean, chain=chain
        )
        center = config.box_center or heme_iron_coordinate(receptor_clean)
        prepare_ligand_sdf(
            ligand_input,
            smiles=config.smiles,
            ligand_sdf=config.ligand_sdf,
            pubchem_cid=config.pubchem_cid,
            compound_name=config.compound_name,
            seed=config.seed,
        )

        with log_path.open("w", encoding="utf-8") as log_handle:
            if config.receptor_pdbqt is not None:
                shutil.copy2(config.receptor_pdbqt, receptor_pdbqt)
                tools["receptor_preparation"] = "preprepared"
            else:
                prep_method = config.receptor_prep
                if prep_method == "auto":
                    try:
                        tools["receptor_preparation"] = _prepare_receptor_pdbqt(
                            receptor_clean,
                            receptor_pdbqt,
                            method="meeko",
                            log_handle=log_handle,
                            cwd=output_dir,
                        )
                    except CYPDockingError as exc:
                        warning = (
                            "Meeko receptor preparation failed for the retained HEM "
                            "cofactor; used Open Babel receptor typing instead. "
                            f"Meeko error: {str(exc).splitlines()[0]}"
                        )
                        warnings.append(warning)
                        log_handle.write(f"FALLBACK: {warning}\n")
                        tools["receptor_preparation"] = _prepare_receptor_pdbqt(
                            receptor_clean,
                            receptor_pdbqt,
                            method="openbabel",
                            log_handle=log_handle,
                            cwd=output_dir,
                        )
                else:
                    tools["receptor_preparation"] = _prepare_receptor_pdbqt(
                        receptor_clean,
                        receptor_pdbqt,
                        method=prep_method,
                        log_handle=log_handle,
                        cwd=output_dir,
                    )
            tools["ligand_preparation"] = _prepare_ligand_pdbqt(
                ligand_input,
                ligand_pdbqt,
                log_handle=log_handle,
                cwd=output_dir,
            )
            vina = _resolve_executable("vina")
            tools["vina"] = vina
            vina_command = [
                vina,
                "--receptor",
                str(receptor_pdbqt),
                "--ligand",
                str(ligand_pdbqt),
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(config.box_size[0]),
                "--size_y",
                str(config.box_size[1]),
                "--size_z",
                str(config.box_size[2]),
                "--exhaustiveness",
                str(config.exhaustiveness),
                "--num_modes",
                str(config.num_modes),
                "--energy_range",
                str(config.energy_range),
                "--cpu",
                str(config.cpu),
                "--seed",
                str(config.seed),
                "--out",
                str(docking_output),
            ]
            _run_command(vina_command, log_handle=log_handle, cwd=output_dir)
            tools["pose_export"] = _export_vina_poses(
                docking_output,
                poses_sdf,
                best_pose_sdf,
                log_handle=log_handle,
                cwd=output_dir,
            )

        scores = parse_vina_scores(docking_output)
        write_scores_csv(scores_csv, scores)
        geometry = closest_ligand_atom_to_heme(receptor_clean, best_pose_sdf)
        write_complex_pdb(receptor_clean, best_pose_sdf, complex_pdb)
        pml_path.write_text(
            build_cyp_pymol_script(
                receptor_pdb=receptor_clean,
                best_pose_sdf=best_pose_sdf,
                output_dir=output_dir,
                closest_atom_rank=int(geometry["atom_rank_for_pymol"]),
                width=config.image_size[0],
                height=config.image_size[1],
            ),
            encoding="utf-8",
        )

        rendered = False
        if config.render:
            pymol = _resolve_executable("pymol", config.pymol_bin)
            tools["pymol"] = pymol
            with log_path.open("a", encoding="utf-8") as log_handle:
                _run_command(
                    [pymol, "-cq", str(pml_path)],
                    log_handle=log_handle,
                    cwd=output_dir,
                )
            expected_images = [
                output_dir / "figure_a_overall_pose.png",
                output_dir / "figure_b_binding_site.png",
                output_dir / "figure_c_heme_geometry.png",
            ]
            if not all(
                path.is_file() and path.stat().st_size > 0
                for path in expected_images
            ):
                raise CYPDockingError("PyMOL completed without all expected PNGs")
            compose_figure(
                output_dir,
                f"{config.isoform} - {config.compound_name} docking hypothesis",
            )
            rendered = True

        result: dict[str, object] = {
            "schema_version": "admet-evo-cyp-docking-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "isoform": config.isoform,
            "compound_name": config.compound_name,
            "receptor": {
                "pdb_id": pdb_id,
                "chain": chain,
                "description": spec.description,
                "cleaning": cleaning,
            },
            "docking": {
                "box_center_angstrom": list(center),
                "box_size_angstrom": list(config.box_size),
                "exhaustiveness": config.exhaustiveness,
                "cpu": config.cpu,
                "num_modes_requested": config.num_modes,
                "seed": config.seed,
                "best_affinity_kcal_mol": scores[0]["affinity_kcal_mol"],
                "poses_found": len(scores),
            },
            "heme_geometry": geometry,
            "rendered": rendered,
            "rendering_recipe": "pymol-figure-rendering/protein-ligand-display",
            "tools": tools,
            "warnings": warnings,
        }
        result["artifacts"] = _artifact_manifest(output_dir)
        _write_json(output_dir / "manifest.json", result)
        return result
    except Exception as exc:
        failure = {
            "schema_version": "admet-evo-cyp-docking-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "isoform": config.isoform,
            "compound_name": config.compound_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "warnings": warnings,
            "artifacts": _artifact_manifest(output_dir),
        }
        _write_json(output_dir / "manifest.json", failure)
        raise


__all__ = [
    "CYP_RECEPTORS",
    "CYPDockingConfig",
    "CYPDockingError",
    "CYPReceptorSpec",
    "build_cyp_pymol_script",
    "clean_cyp_receptor",
    "closest_ligand_atom_to_heme",
    "heme_iron_coordinate",
    "normalize_isoform",
    "parse_vina_scores",
    "run_cyp_docking",
    "validate_config",
]
