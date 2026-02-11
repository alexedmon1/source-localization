"""HTML Report Generation for Pipeline Results."""

import base64
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_b64


def generate_pipeline_report(config, step_outputs, output_path):
    """
    Generate comprehensive HTML report of pipeline results.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    step_outputs : dict
        Outputs from all pipeline steps
    output_path : Path
        Path to save HTML report
    """
    html_sections = []

    # Header
    html_sections.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Source Localization Pipeline Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #555;
        }}
        .step-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .status-complete {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 15px 0;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 5px 15px 5px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Source Localization Pipeline Report</h1>
    <div class="metadata">
        <p><span class="metric-label">Pipeline:</span> {config['pipeline']['name']}</p>
        <p><span class="metric-label">Date:</span> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><span class="metric-label">BEM Type:</span> {config['pipeline']['bem_type']}</p>
        <p><span class="metric-label">Source Type:</span> {config['pipeline']['source_type']}</p>
        <p><span class="metric-label">Inverse Method:</span> {config['inverse']['method']}</p>
        <p><span class="metric-label">Output Directory:</span> {config['outputs']['dir']}</p>
    </div>
""")

    # Step 1: Electrode Registration
    if 'electrode_registration' in step_outputs:
        html_sections.append(_generate_electrode_section(step_outputs['electrode_registration'], config))

    # Step 2: EEG Data
    if 'eeg_data' in step_outputs:
        html_sections.append(_generate_eeg_data_section(step_outputs['eeg_data']))

    # Step 3: BEM Model
    if 'bem_model' in step_outputs:
        html_sections.append(_generate_bem_section(step_outputs['bem_model'], config))

    # Step 4: Source Space
    if 'source_space' in step_outputs:
        html_sections.append(_generate_source_space_section(step_outputs['source_space'], config))

    # Step 5: Forward Solution
    if 'forward_solution' in step_outputs:
        html_sections.append(_generate_forward_section(step_outputs['forward_solution'], config))

    # Step 6: Inverse Solution
    if 'inverse_solution' in step_outputs:
        html_sections.append(_generate_inverse_section(step_outputs['inverse_solution'], config))

    # Step 7: ROI Extraction
    if 'roi_extraction' in step_outputs:
        html_sections.append(_generate_roi_section(step_outputs['roi_extraction'], config))

    # Step 8: Spectral Analysis
    if 'spectral_analysis' in step_outputs:
        html_sections.append(_generate_spectral_section(step_outputs['spectral_analysis'], config))

    # Step 9: Visualization
    if 'visualization' in step_outputs:
        html_sections.append(_generate_visualization_section(step_outputs['visualization'], config))

    # Validation Results (if available)
    html_sections.append(_generate_validation_section(config))

    # Footer
    html_sections.append("""
</div>
</body>
</html>
""")

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_sections))


def _generate_electrode_section(outputs, config):
    """Generate HTML section for electrode registration step."""
    info = outputs['info']
    electrode_results = outputs.get('electrode_results', {})
    n_electrodes = len(info['ch_names'])

    # Check for figure
    output_dir = Path(config['outputs']['dir'])
    fig_path = output_dir / 'figures' / 'step1_electrodes.png'
    fig_html = f'<img src="figures/step1_electrodes.png" alt="Electrode Registration">' if fig_path.exists() else ''

    bregma_val = electrode_results.get('bregma_validation', {})
    validation_status = "PASSED" if bregma_val.get('lambda_all_checks_pass', False) else "FAILED"

    return f"""
    <div class="step-section">
        <h2>Step 1: Electrode Registration</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Electrodes:</span> {n_electrodes}</p>
        <p><span class="metric-label">Projection Method:</span> {config['electrode'].get('projection_method', 'N/A')}</p>
        <p><span class="metric-label">Bregma-Lambda Validation:</span> {validation_status}</p>
        {fig_html}
    </div>
"""


def _generate_eeg_data_section(outputs):
    """Generate HTML section for EEG data loading step."""
    epochs = outputs.get('epochs')
    if epochs is None:
        return ""

    return f"""
    <div class="step-section">
        <h2>Step 2: EEG Data Loading</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Epochs:</span> {len(epochs)}</p>
        <p><span class="metric-label">Sampling Rate:</span> {epochs.info['sfreq']} Hz</p>
        <p><span class="metric-label">Channels:</span> {len(epochs.ch_names)}</p>
        <p><span class="metric-label">Time Range:</span> [{epochs.times[0]:.3f}, {epochs.times[-1]:.3f}] s</p>
    </div>
"""


def _generate_source_space_section(outputs, config=None):
    """Generate HTML section for source space step."""
    source_coords_mm = outputs['source_coords_mm']
    n_sources = outputs['n_sources']
    source_type = outputs['source_type']

    # Check for figure
    fig_html = ''
    if config:
        output_dir = Path(config['outputs']['dir'])
        fig_path = output_dir / 'figures' / 'step3_source_space.png'
        if fig_path.exists():
            fig_html = '<img src="figures/step3_source_space.png" alt="Source Space">'
    else:
        fig_html = '<img src="figures/step3_source_space.png" alt="Source Space">'

    return f"""
    <div class="step-section">
        <h2>Step 4: Source Space</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Type:</span> {source_type}</p>
        <p><span class="metric-label">Sources:</span> {n_sources:,}</p>
        <p><span class="metric-label">Coordinate Range (mm):</span><br>
           X: [{source_coords_mm[:, 0].min():.2f}, {source_coords_mm[:, 0].max():.2f}]<br>
           Y: [{source_coords_mm[:, 1].min():.2f}, {source_coords_mm[:, 1].max():.2f}]<br>
           Z: [{source_coords_mm[:, 2].min():.2f}, {source_coords_mm[:, 2].max():.2f}]</p>
        {fig_html}
    </div>
"""


def _generate_forward_section(outputs, config=None):
    """Generate HTML section for forward solution step."""
    fwd = outputs['fwd']

    # Check for figure
    fig_html = ''
    if config:
        output_dir = Path(config['outputs']['dir'])
        fig_path = output_dir / 'figures' / 'step4_forward.png'
        if fig_path.exists():
            fig_html = '<img src="figures/step4_forward.png" alt="Forward Solution">'
    else:
        fig_html = '<img src="figures/step4_forward.png" alt="Forward Solution">'

    return f"""
    <div class="step-section">
        <h2>Step 5: Forward Solution</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Channels:</span> {fwd['nchan']}</p>
        <p><span class="metric-label">Sources:</span> {fwd['nsource']}</p>
        {fig_html}
    </div>
"""


def _generate_inverse_section(outputs, config):
    """Generate HTML section for inverse solution step."""
    stc = outputs['stc']
    method = outputs['method']

    # Check for magnitude and signed figures
    output_dir = Path(config['outputs']['dir'])
    mag_fig = output_dir / 'figures' / 'step5_inverse_magnitude.png'
    signed_fig = output_dir / 'figures' / 'step5_inverse_signed.png'
    old_fig = output_dir / 'figures' / 'step5_inverse.png'

    fig_html = ''
    if mag_fig.exists():
        fig_html += '<h4>Magnitude (Unsigned)</h4>\n'
        fig_html += '<img src="figures/step5_inverse_magnitude.png" alt="Inverse Solution (Magnitude)">\n'
    if signed_fig.exists():
        fig_html += '<h4>Signed</h4>\n'
        fig_html += '<img src="figures/step5_inverse_signed.png" alt="Inverse Solution (Signed)">\n'
    if not fig_html and old_fig.exists():
        fig_html = '<img src="figures/step5_inverse.png" alt="Inverse Solution">'

    return f"""
    <div class="step-section">
        <h2>Step 6: Inverse Solution</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Method:</span> {method}</p>
        <p><span class="metric-label">SNR:</span> {config['inverse']['snr']}</p>
        <p><span class="metric-label">Sources:</span> {len(stc.data)}</p>
        <p><span class="metric-label">Time Points:</span> {stc.data.shape[1]}</p>
        {fig_html}
    </div>
"""


def _generate_roi_section(outputs, config=None):
    """Generate HTML section for ROI extraction step."""
    roi_labels = outputs['roi_labels']
    roi_source_mapping = outputs['roi_source_mapping']

    sources_per_roi = {roi: len(sources) for roi, sources in roi_source_mapping.items()}
    rois_with_sources = sum(1 for n in sources_per_roi.values() if n > 0)

    # Check for magnitude and signed figures
    fig_html = ''
    if config:
        output_dir = Path(config['outputs']['dir'])
        mag_fig = output_dir / 'figures' / 'step6_roi_extraction_magnitude.png'
        signed_fig = output_dir / 'figures' / 'step6_roi_extraction_signed.png'
        old_fig = output_dir / 'figures' / 'step6_roi_extraction.png'

        if mag_fig.exists():
            fig_html += '<h4>Magnitude (Unsigned)</h4>\n'
            fig_html += '<img src="figures/step6_roi_extraction_magnitude.png" alt="ROI Extraction (Magnitude)">\n'
        if signed_fig.exists():
            fig_html += '<h4>Signed</h4>\n'
            fig_html += '<img src="figures/step6_roi_extraction_signed.png" alt="ROI Extraction (Signed)">\n'
        if not fig_html and old_fig.exists():
            fig_html = '<img src="figures/step6_roi_extraction.png" alt="ROI Extraction">'
    else:
        fig_html = '<img src="figures/step6_roi_extraction.png" alt="ROI Extraction">'

    return f"""
    <div class="step-section">
        <h2>Step 7: ROI Extraction</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Total ROIs:</span> {len(roi_labels)}</p>
        <p><span class="metric-label">ROIs with Sources:</span> {rois_with_sources}</p>
        {fig_html}
    </div>
"""


def _generate_spectral_section(outputs, config):
    """Generate HTML section for spectral analysis step."""
    roi_band_power = outputs['roi_band_power']
    freq_bands = config['spectral']['frequency_bands']

    # Check for figure
    fig_html = ''
    output_dir = Path(config['outputs']['dir'])
    fig_path = output_dir / 'figures' / 'step7_band_power.png'
    if fig_path.exists():
        fig_html = '<img src="figures/step7_band_power.png" alt="Band Power">'

    band_table = "<table><tr><th>Band</th><th>Range (Hz)</th></tr>"
    for band_name, band_range in freq_bands.items():
        band_table += f"<tr><td>{band_name}</td><td>{band_range[0]}-{band_range[1]}</td></tr>"
    band_table += "</table>"

    return f"""
    <div class="step-section">
        <h2>Step 8: Spectral Analysis</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Frequency Bands:</span></p>
        {band_table}
        {fig_html}
    </div>
"""


def _generate_visualization_section(outputs, config):
    """Generate HTML section for visualization step."""
    figure_paths = outputs.get('figure_paths', [])

    return f"""
    <div class="step-section">
        <h2>Step 9: Visualization</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Figures Generated:</span> {len(figure_paths)}</p>
    </div>
"""


def _generate_bem_section(outputs, config):
    """Generate HTML section for BEM model step."""
    bem_type = config['pipeline']['bem_type']
    bem_params = outputs.get('bem_params', {})

    # Extract BEM details
    n_layers = bem_params.get('n_layers', config['bem'][bem_type].get('n_layers', 3))
    conductivities = bem_params.get('conductivities', config['bem'][bem_type].get('conductivities', []))

    # BEM geometry details
    center = bem_params.get('center', None)
    semi_axes = bem_params.get('semi_axes', None)
    radius = bem_params.get('radius', None)

    geometry_html = ''
    if bem_type == 'ellipsoid' and semi_axes is not None:
        geometry_html = f"""
        <p><span class="metric-label">Semi-axes (mm):</span> [{semi_axes[0]:.2f}, {semi_axes[1]:.2f}, {semi_axes[2]:.2f}]</p>
        """
    elif bem_type == 'sphere' and radius is not None:
        geometry_html = f"""
        <p><span class="metric-label">Radius (mm):</span> {radius:.2f}</p>
        """

    if center is not None:
        geometry_html += f"""
        <p><span class="metric-label">Center (mm):</span> [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]</p>
        """

    conductivity_html = ''
    if conductivities:
        conductivity_html = f"""
        <p><span class="metric-label">Conductivities (S/m):</span> {', '.join([f'{c:.4f}' for c in conductivities])}</p>
        """

    return f"""
    <div class="step-section">
        <h2>Step 3: BEM Model</h2>
        <p class="status-complete">✓ Complete</p>
        <p><span class="metric-label">Type:</span> {bem_type}</p>
        <p><span class="metric-label">Layers:</span> {n_layers}</p>
        {geometry_html}
        {conductivity_html}
    </div>
"""


def _generate_validation_section(config):
    """Generate HTML section for validation results with localization error maps."""
    output_dir = Path(config['outputs']['dir'])
    figures_dir = output_dir / 'figures'

    # Look for localization error map figures
    error_map_files = list(figures_dir.glob('localization_error_map*.png'))
    validated_error_files = list(figures_dir.glob('validated_error_map*.png'))

    if not error_map_files and not validated_error_files:
        return ""

    fig_html = ''

    # Show validated error maps first (from actual dipole simulation)
    for fig_file in sorted(validated_error_files):
        fig_html += f'<h4>{fig_file.stem}</h4>\n'
        fig_html += f'<img src="figures/{fig_file.name}" alt="Validated Localization Error Map">\n'

    # Show estimated error maps
    for fig_file in sorted(error_map_files):
        fig_html += f'<h4>{fig_file.stem}</h4>\n'
        fig_html += f'<img src="figures/{fig_file.name}" alt="Localization Error Map">\n'

    # Look for validation results JSON
    validation_html = ''
    validation_json = output_dir / 'validation_results.json'
    if validation_json.exists():
        try:
            with open(validation_json) as f:
                results = json.load(f)

            if isinstance(results, dict):
                mean_error = results.get('mean_error_mm', results.get('mean_localization_error_mm'))
                median_error = results.get('median_error_mm', results.get('median_localization_error_mm'))
                roi_accuracy = results.get('roi_accuracy', results.get('roi_classification_accuracy'))

                if mean_error is not None:
                    validation_html += f'<p><span class="metric-label">Mean Localization Error:</span> {mean_error:.2f} mm</p>\n'
                if median_error is not None:
                    validation_html += f'<p><span class="metric-label">Median Localization Error:</span> {median_error:.2f} mm</p>\n'
                if roi_accuracy is not None:
                    validation_html += f'<p><span class="metric-label">ROI Classification Accuracy:</span> {roi_accuracy*100:.1f}%</p>\n'
        except Exception:
            pass

    return f"""
    <div class="step-section">
        <h2>Validation Results</h2>
        <p class="status-complete">✓ Localization Error Analysis</p>
        {validation_html}
        {fig_html}
    </div>
"""
