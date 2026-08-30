"""Visual output: the bullseye plot and the per-frame tracking overlays.

The bullseye shows the spin axis in the eY-eZ plane. The centre is pure
backspin - axis perfectly horizontal and perpendicular to the shot direction -
so radial distance measures how far the release deviates from ideal, and the
coloured bands turn that into a verdict. With more than one shot, the mean
point and the spread region appear as well.

`save_track_visualizations` saves, for every analysed frame, the full view and
a magnified crop, to inspect at pixel level which dots were matched together.
"""

import os

import cv2
import numpy as np

from spindoctor.config import ZOOM_FACTOR, ZOOM_PADDING


# --- Bullseye palette ---
# The band colours are a STATUS scale (ideal -> poor), not an identity one:
# they are fixed and must not be reused for data. Shots use a categorical blue,
# distinct from all four bands, so a shot is never mistaken for a zone.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
HAIRLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
SERIES = "#2a78d6"

# Quality bands: (outer radius, status colour, label)
BANDS = [
    (0.2, "#0ca30c", "Ideal"),
    (0.4, "#fab219", "Good"),
    (0.6, "#ec835a", "Fair"),
    (0.8, "#d03b3b", "Poor"),
]
BAND_ALPHA = 0.13   # large fills stay washes, never saturated blocks

AXIS_LIMIT = 0.8
FONT_STACK = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def bullseye_color_for_radius(r):
    """Quality band for a radial distance in the eY-eZ plane."""
    for outer, color, label in BANDS:
        if r <= outer:
            return label, color
    return BANDS[-1][2], BANDS[-1][1]


def draw_bullseye(ax):
    """Draw the target: status bands, axis crosshair, ticks."""
    from matplotlib.patches import Wedge
    import matplotlib.pyplot as plt

    # Rings, not stacked discs: with discs the alphas compound and the green
    # "Ideal" centre turns olive, no longer matching its legend swatch.
    inner = 0.0
    for outer, color, _ in BANDS:
        ax.add_patch(Wedge((0, 0), outer, 0, 360, width=outer - inner,
                           facecolor=color, alpha=BAND_ALPHA,
                           edgecolor="none", zorder=0))
        inner = outer
    # A thin boundary per band, to read the steps without adding weight.
    for outer, color, _ in BANDS:
        ax.add_patch(plt.Circle((0, 0), outer, facecolor="none",
                                edgecolor=color, alpha=0.35, linewidth=0.8, zorder=1))

    # Solid hairline, never dashed: dashing would read as a threshold or a
    # projection, and this is only a reference.
    ax.axhline(0, color=AXIS_RULE, linewidth=1.0, zorder=2)
    ax.axvline(0, color=AXIS_RULE, linewidth=1.0, zorder=2)

    # No radial labels on the rings: the band legend below and the axis ticks
    # already give the radius, and repeating it would clutter the crosshair.

    ax.set_xlim(AXIS_LIMIT, -AXIS_LIMIT)   # +eZ to the left
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal", "box")

    ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0, pad=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_STACK)

    for spine in ax.spines.values():
        spine.set_edgecolor(HAIRLINE)
        spine.set_linewidth(1.0)

    ax.set_xlabel("eZ", fontsize=9, color=INK_SECONDARY, labelpad=6,
                  fontfamily=FONT_STACK)
    ax.set_ylabel("eY", fontsize=9, color=INK_SECONDARY, labelpad=6,
                  fontfamily=FONT_STACK)
    ax.set_facecolor(SURFACE)


def _stat(fig, x, y, value, label, value_size=15, value_color=INK_PRIMARY):
    """One column of the stat row: value above, label below."""
    fig.text(x, y, value, fontsize=value_size, fontweight="semibold",
             color=value_color, ha="left", va="baseline", fontfamily=FONT_STACK)
    fig.text(x, y - 0.028, label, fontsize=8, color=INK_MUTED,
             ha="left", va="baseline", fontfamily=FONT_STACK)


def make_bullseye(ez_vals, ey_vals, ex_vals, out_png, title="Bullseye",
                  num_tracked_dots=None, spin_rate_hz=None):
    """
    Bullseye plot of the spin axis in the eY-eZ plane.

    Args:
        ez_vals, ey_vals, ex_vals: Spin axis components, one entry per shot
        out_png: Path of the PNG to write
        title: Title (clip or player name)
        num_tracked_dots: Number of tracked dots, if known
        spin_rate_hz: Spin rate for the stat row, if known
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Patch
    except ImportError:
        return

    n = len(ez_vals)
    if n == 0:
        return

    fig = plt.figure(figsize=(6.0, 6.9), dpi=200, facecolor=SURFACE)

    # --- Header ---------------------------------------------------------------
    fig.text(0.08, 0.955, title, fontsize=14, fontweight="semibold",
             color=INK_PRIMARY, ha="left", va="baseline", fontfamily=FONT_STACK)
    shots_label = "1 shot" if n == 1 else f"{n} shots"
    fig.text(0.08, 0.930, f"Spin axis in the eY-eZ plane  ·  {shots_label}",
             fontsize=9, color=INK_MUTED, ha="left", va="baseline",
             fontfamily=FONT_STACK)

    # --- Metrics --------------------------------------------------------------
    ez_avg = float(np.mean(ez_vals))
    ey_avg = float(np.mean(ey_vals))
    ex_avg = float(np.mean(ex_vals))

    r_mean = np.sqrt(ey_avg ** 2 + ez_avg ** 2)
    rq_pct = abs(ex_avg) * 100.0
    band_label, band_color = bullseye_color_for_radius(r_mean)

    # Release quality leads; the rest supports it. The number stays in primary
    # ink and the status is carried by the coloured dot beside the label, never
    # by the colour of the text itself.
    _stat(fig, 0.08, 0.860, f"{rq_pct:.1f}%", "Release quality", value_size=26)
    fig.add_artist(Line2D([0.238], [0.838], marker="o", linestyle="none",
                          markersize=7, color=band_color,
                          transform=fig.transFigure))
    fig.text(0.252, 0.832, band_label, fontsize=9, color=INK_SECONDARY,
             ha="left", va="baseline", fontfamily=FONT_STACK)

    col_x = 0.56
    if spin_rate_hz is not None:
        _stat(fig, col_x, 0.860, f"{spin_rate_hz:.1f} Hz", "Spin rate")
        col_x += 0.19
    misalign_deg = np.degrees(np.arccos(np.clip(abs(ex_avg), -1.0, 1.0)))
    _stat(fig, col_x, 0.860, f"{misalign_deg:.1f}°", "Misalignment")

    fig.add_artist(Line2D([0.08, 0.92], [0.806, 0.806], color=HAIRLINE,
                          linewidth=1.0, transform=fig.transFigure))

    # --- Target ---------------------------------------------------------------
    ax = fig.add_axes([0.13, 0.255, 0.74, 0.525])
    draw_bullseye(ax)

    ez_plot = [-v for v in ez_vals]        # +eZ to the left
    ez_avg_plot = -ez_avg

    sd_radius = None
    if n > 1:
        # Spread: RMS radius around the mean point.
        mean_r2 = sum((ey_vals[i] - ey_avg) ** 2 + (ez_vals[i] - ez_avg) ** 2
                      for i in range(n)) / n
        sd_radius = float(np.sqrt(mean_r2))
        # Wash plus solid edge, so it reads as a region - which it is.
        ax.add_patch(Circle((ez_avg_plot, ey_avg), sd_radius, facecolor=SERIES,
                            alpha=0.10, edgecolor="none", zorder=3))
        ax.add_patch(Circle((ez_avg_plot, ey_avg), sd_radius, facecolor="none",
                            edgecolor=SERIES, alpha=0.55, linewidth=1.5, zorder=4))

    # A surface ring keeps the dots legible where they overlap each other or
    # a band boundary.
    ax.scatter(ez_plot, ey_vals, s=46, color=SERIES, alpha=0.85,
               edgecolor=SURFACE, linewidth=1.5, zorder=5)

    if n > 1:
        ax.scatter([ez_avg_plot], [ey_avg], s=170, color=SERIES,
                   edgecolor=SURFACE, linewidth=2.0, zorder=6)

    # --- Legends --------------------------------------------------------------
    # One shot needs no marker legend: the title already says what is drawn. The
    # bands are always labelled, because a status colour must never carry
    # meaning on its own.
    if n > 1:
        marker_handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=SERIES, markeredgecolor=SURFACE,
                   markeredgewidth=1.5, label="Shot"),
            Line2D([], [], marker="o", linestyle="none", markersize=10,
                   markerfacecolor=SERIES, markeredgecolor=SURFACE,
                   markeredgewidth=2.0, label="Mean"),
            Line2D([], [], marker="o", linestyle="none", markersize=10,
                   markerfacecolor=(0.165, 0.471, 0.839, 0.10),
                   markeredgecolor=SERIES, markeredgewidth=1.5, label="Spread (RMS)"),
        ]
        leg = ax.legend(handles=marker_handles, loc="upper center",
                        bbox_to_anchor=(0.5, -0.105), ncol=3, frameon=False,
                        handletextpad=0.5, columnspacing=1.8, fontsize=8.5)
        for text in leg.get_texts():
            text.set_color(INK_SECONDARY)
            text.set_fontfamily(FONT_STACK)
        ax.add_artist(leg)
        band_anchor = -0.185
    else:
        band_anchor = -0.105

    band_handles = [Patch(facecolor=color, alpha=BAND_ALPHA,
                          edgecolor=color, linewidth=0.8,
                          label=f"{label} ≤ {outer:.1f}" if outer < AXIS_LIMIT
                          else f"{label} > {BANDS[-2][0]:.1f}")
                    for outer, color, label in BANDS]
    band_leg = ax.legend(handles=band_handles, loc="upper center",
                         bbox_to_anchor=(0.5, band_anchor), ncol=4, frameon=False,
                         handletextpad=0.5, columnspacing=1.2, fontsize=8.5)
    for text in band_leg.get_texts():
        text.set_color(INK_SECONDARY)
        text.set_fontfamily(FONT_STACK)

    # --- Footer ---------------------------------------------------------------
    notes = ["Center = pure backspin; radius = deviation from ideal"]
    if sd_radius is not None:
        notes.append(f"Release consistency (SD area): {np.pi * sd_radius ** 2:.4f}")
    if num_tracked_dots is not None:
        notes.append(f"Tracked dots: {num_tracked_dots}")
    fig.text(0.08, 0.052, "\n".join(notes), fontsize=8, color=INK_MUTED,
             ha="left", va="baseline", linespacing=1.6, fontfamily=FONT_STACK)

    # No tight_layout or bbox_inches: the canvas is at fixed coordinates, so
    # the result is identical on every run.
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_aggregated_bullseye(player_results, output_path, player_name):
    """
    Create aggregated bullseye plot for a player using all their video results.

    Args:
        player_results: List of result dicts from process_video
        output_path: Path to save the aggregated bullseye PNG
        player_name: Name of the player for the title
    """
    if not player_results:
        return

    ez_vals = [r['ez'] for r in player_results]
    ey_vals = [r['ey'] for r in player_results]
    ex_vals = [r['ex'] for r in player_results]

    rates = [r['spin_rate_hz'] for r in player_results if r.get('spin_rate_hz')]

    make_bullseye(ez_vals, ey_vals, ex_vals, output_path,
                  title=player_name,
                  spin_rate_hz=float(np.mean(rates)) if rates else None)


def save_track_visualizations(sorted_frames, all_detections, final_tracks, video_output_folder):
    """
    Save the tracking visualisation for each analysed frame:
    - track_viz_<frame>.jpg: full frame with ball, dots and numbered tracks
    - track_viz_<frame>_ZOOM.jpg: magnified crop for pixel-level inspection
    """
    # Draw tracks on frames
    for f_idx in sorted_frames:
        img = all_detections[f_idx]['frame'].copy()
        data = all_detections[f_idx]

        # Draw Ball
        cv2.circle(img, data['ball_center'], int(data['ball_radius']), (255, 0, 0), 2)

        # Draw Dots (All)
        for d in data['dots']:
            cv2.circle(img, (d[0], d[1]), 4, (100, 100, 100), -1)

        # Draw Tracked Dots
        for t_id, track in enumerate(final_tracks):
            if f_idx in track:
                d_idx = track[f_idx]
                dx, dy, _, _ = data['dots'][d_idx]
                color = ((t_id * 50) % 255, (t_id * 120) % 255, (t_id * 200 + 100) % 255)
                cv2.circle(img, (dx, dy), 8, color, -1)
                cv2.putText(img, str(t_id), (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imwrite(os.path.join(video_output_folder, f"track_viz_{f_idx}.jpg"), img)

        # --- ZOOM VISUALIZATION: Save zoomed view for pixel-level inspection ---
        # This allows verification of dot selection accuracy at the pixel level
        ball_box = data['ball_box']
        x1, y1, x2, y2 = ball_box
        ball_w = x2 - x1
        ball_h = y2 - y1
        pad_w = ball_w * (ZOOM_PADDING - 1.0) / 2.0
        pad_h = ball_h * (ZOOM_PADDING - 1.0) / 2.0
        crop_x1 = int(max(0, x1 - pad_w))
        crop_y1 = int(max(0, y1 - pad_h))
        crop_x2 = int(min(img.shape[1], x2 + pad_w))
        crop_y2 = int(min(img.shape[0], y2 + pad_h))

        # Crop and zoom the visualization (with tracks drawn)
        crop_viz = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_viz.size > 0:
            zoom_h, zoom_w = int(crop_viz.shape[0] * ZOOM_FACTOR), int(crop_viz.shape[1] * ZOOM_FACTOR)
            zoomed_viz = cv2.resize(crop_viz, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

            # Add zoom info text
            cv2.putText(zoomed_viz, f"ZOOM: {ZOOM_FACTOR}x", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(zoomed_viz, f"Frame: {f_idx}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(video_output_folder, f"track_viz_{f_idx}_ZOOM.jpg"), zoomed_viz)
