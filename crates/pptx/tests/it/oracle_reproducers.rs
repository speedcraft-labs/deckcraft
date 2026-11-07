//! Regression checks for the external LibreOffice oracle reproducers.
//! Set `ARTCRAFT_CORPUS` to the corpus root to run these against the cited PPTX files.

use std::path::PathBuf;

use deckcraft_model::{Shape, ShapeKind};

fn fixture(path: &str) -> Option<deckcraft_model::Presentation> {
    let root = std::env::var_os("ARTCRAFT_CORPUS").map(PathBuf::from)?;
    let path = root.join("office-libreoffice").join(path);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Some(deckcraft_pptx::import(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

fn has_diagram(shapes: &[Shape]) -> bool {
    shapes.iter().any(|s| {
        (s.diagram.is_some() && matches!(&s.kind, ShapeKind::Group { children, .. } if !children.is_empty()))
            || s.children().iter().any(|c| has_diagram(std::slice::from_ref(c)))
    })
}

fn charts(shapes: &[Shape], out: &mut Vec<deckcraft_model::chart::Chart>) {
    for s in shapes {
        if let ShapeKind::Chart(c) = &s.kind {
            out.push((**c).clone());
        }
        charts(s.children(), out);
    }
}

#[test]
fn cited_smartart_files_import_the_cached_drawing_before_any_layout_work() {
    let Some(_) = std::env::var_os("ARTCRAFT_CORPUS") else { return };
    for path in [
        "sd/qa/unit/data/pptx/table-list.pptx",
        "sd/qa/unit/data/pptx/smartart-composite-infer-right.pptx",
        "sd/qa/unit/data/pptx/smartart-org-chart2.pptx",
        "sd/qa/unit/data/pptx/smartart-vertical-block-list.pptx",
        "sd/qa/unit/data/pptx/smartart-snake-rows.pptx",
        "sd/qa/unit/data/pptx/vertical-bracket-list.pptx",
    ] {
        let p = fixture(path).expect("ARTCRAFT_CORPUS set");
        assert!(p.slides.iter().any(|slide| has_diagram(&slide.shapes)), "{path}: SmartArt fallback not imported");
    }
}

#[test]
fn cited_chart_files_keep_custom_labels_and_3d_plot_kind() {
    if std::env::var_os("ARTCRAFT_CORPUS").is_none() {
        return;
    }
    let p = fixture("chart2/qa/extras/data/pptx/tdf122765.pptx").expect("ARTCRAFT_CORPUS set");
    let mut found = Vec::new();
    for slide in &p.slides {
        charts(&slide.shapes, &mut found);
    }
    assert!(
        found.iter().any(|c| c.series.iter().any(|s| s.show_category_name == Some(true) && s.show_percent == Some(true))),
        "pie category/percent labels missing"
    );

    let p = fixture("chart2/qa/extras/xshape/data/pptx/tdf149204.pptx").expect("ARTCRAFT_CORPUS set");
    let mut found = Vec::new();
    for slide in &p.slides {
        charts(&slide.shapes, &mut found);
    }
    assert!(found.iter().any(|c| c.is_3d), "3-D plot kind lost during import");

    let p = fixture("chart2/qa/extras/data/pptx/CustomDataLabel_tdf115107.pptx").expect("ARTCRAFT_CORPUS set");
    let mut found = Vec::new();
    for slide in &p.slides {
        charts(&slide.shapes, &mut found);
    }
    let labels = found.iter().flat_map(|c| c.series.iter()).find(|s| s.custom_labels.iter().any(Option::is_some)).map(|s| &s.custom_labels);
    let labels = labels.expect("explicit point labels imported");
    assert_eq!(labels.first().and_then(Option::as_deref), Some("90.0 = 90"));
    assert_eq!(labels.get(1).and_then(Option::as_deref), Some("Text : B\nMultiline\nAbc"));
    assert_eq!(labels.get(2).and_then(Option::as_deref), Some("DATA"));
    assert_eq!(labels.get(3).and_then(Option::as_deref), Some("70 <CELLREF"));
}
