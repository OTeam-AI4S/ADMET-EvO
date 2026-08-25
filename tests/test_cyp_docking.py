from pathlib import Path

import pytest

from admet_agent.cyp_docking import (
    CYPDockingConfig,
    build_cyp_pymol_script,
    clean_cyp_receptor,
    heme_iron_coordinate,
    normalize_isoform,
    parse_vina_scores,
    validate_config,
)


def _pdb_line(
    record: str,
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:>5} {atom:^4} {residue:>3} {chain:1}"
        f"{residue_number:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"  1.00 20.00          {element:>2}"
    )


def test_normalize_isoform_accepts_short_and_canonical_names():
    assert normalize_isoform("3a4") == "CYP3A4"
    assert normalize_isoform("cyp-2d6") == "CYP2D6"
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_isoform("CYP9Z9")


def test_validate_config_requires_exactly_one_ligand_source(tmp_path):
    base = CYPDockingConfig(isoform="3A4", output_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="exactly one"):
        validate_config(base)
    with pytest.raises(ValueError, match="exactly one"):
        validate_config(
            CYPDockingConfig(
                isoform="3A4",
                output_dir=tmp_path / "run",
                smiles="CCO",
                pubchem_cid=702,
            )
        )


def test_validate_config_accepts_auditable_auto_receptor_fallback(tmp_path):
    config = validate_config(
        CYPDockingConfig(
            isoform="CYP3A4",
            output_dir=tmp_path / "run",
            smiles="CCO",
        )
    )
    assert config.receptor_prep == "auto"


def test_validate_config_rejects_whitespace_in_output_path(tmp_path):
    with pytest.raises(ValueError, match="whitespace"):
        validate_config(
            CYPDockingConfig(
                isoform="CYP3A4",
                output_dir=tmp_path / "run with spaces",
                smiles="CCO",
            )
        )


def test_clean_receptor_preserves_chain_heme_and_removes_other_hetero(tmp_path):
    source = tmp_path / "source.pdb"
    source.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 1, 2, 3, "C"),
                _pdb_line("ATOM", 2, "CA", "GLY", "B", 1, 3, 2, 1, "C"),
                _pdb_line("HETATM", 3, "FE", "HEM", "A", 500, 4, 5, 6, "FE"),
                _pdb_line("HETATM", 4, "C1", "LIG", "A", 600, 7, 8, 9, "C"),
                _pdb_line("HETATM", 5, "O", "HOH", "A", 700, 1, 1, 1, "O"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cleaned = tmp_path / "cleaned.pdb"
    stats = clean_cyp_receptor(source, cleaned, chain="A")
    text = cleaned.read_text(encoding="utf-8")
    assert "ALA" in text
    assert "HEM" in text
    assert "GLY" not in text
    assert "LIG" not in text
    assert "HOH" not in text
    assert stats["heme_atoms"] == 1
    assert heme_iron_coordinate(cleaned) == pytest.approx((4.0, 5.0, 6.0))


def test_clean_receptor_fails_closed_without_heme(tmp_path):
    source = tmp_path / "source.pdb"
    source.write_text(
        _pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 1, 2, 3, "C") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="No HEM"):
        clean_cyp_receptor(source, tmp_path / "cleaned.pdb", chain="A")


def test_parse_vina_scores_reads_all_modes(tmp_path):
    result = tmp_path / "poses.pdbqt"
    result.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT:    -8.700      0.000      0.000\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK VINA RESULT:    -8.100      1.250      2.300\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    scores = parse_vina_scores(result)
    assert [row["mode"] for row in scores] == [1, 2]
    assert scores[0]["affinity_kcal_mol"] == pytest.approx(-8.7)
    assert scores[1]["rmsd_ub_angstrom"] == pytest.approx(2.3)


def test_pymol_script_preserves_heme_and_rays_before_each_png(tmp_path):
    script = build_cyp_pymol_script(
        receptor_pdb=Path("receptor.pdb"),
        best_pose_sdf=Path("pose.sdf"),
        output_dir=tmp_path,
        closest_atom_rank=4,
        width=1200,
        height=900,
    )
    assert "create heme, source_receptor and resn HEM" in script
    assert "contact_residues, byres (receptor within 4.0 of ligand)" in script
    assert "ligand and rank 4" in script
    assert script.count("ray 1200, 900\npng") == 3
    assert "save " in script and "pymol_docking_session.pse" in script
