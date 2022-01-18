# vim: fdm=indent
'''
author:     Fabio Zanini
date:       19/05/21
content:    8 slices figure with offset center
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':


    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('fold change (kids)')
    ax.set_ylabel('fold change (adults)')


    def draw_8_slices(ax, center, radius=1.0):
        def cp(amin, amax, center, ndots=100):
            xc, yc = center
            angles = np.linspace(amin, amax, ndots)
            xcircs, ycircs = xc + radius * np.cos(angles), yc + radius * np.sin(angles)
            return [(xi, yi) for xi, yi in zip(xcircs, ycircs)][::-1]

        def draw_wedge(ax, points, color):
            points = np.array(points)
            center = points.mean(axis=0)
            delta = points - center
            points = center + 0.99 * delta

            rgb = list(mpl.colors.to_rgba(color)[:-1])
            rgba_edge = tuple(rgb + [0.6])
            rgba_fill = tuple(rgb + [0.2])
            ax.add_artist(plt.Polygon(
                points, edgecolor=rgba_edge, facecolor=rgba_fill, lw=2,
                clip_on=False,
                ))

        xc, yc = center
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Cross
        ax.axvline(xc, ls='--', color='k')
        ax.axhline(yc, ls='--', color='k')

        # Circle
        circle = plt.Circle(center, radius, facecolor='none', edgecolor='k')
        ax.add_artist(circle)

        # Wedges
        # 1. intersections with circle
        angles = 2 * np.pi / 8 * (np.arange(8) + 0.5)
        xcircs, ycircs = xc + radius * np.cos(angles), yc + radius * np.sin(angles)
        circs = np.vstack([xcircs, ycircs]).T
        # 2. Intersections with axes border
        xbords, ybords = [], []
        xbords.append(xmax)
        ybords.append(yc + (ycircs[0] - yc) / (xcircs[0] - xc) * (xmax - xc))
        xbords.append(xc + (xcircs[1] - xc) / (ycircs[1] - yc) * (ymax - yc))
        ybords.append(ymax)
        xbords.append(xc + (xcircs[2] - xc) / (ycircs[2] - yc) * (ymax - yc))
        ybords.append(ymax)
        xbords.append(xmin)
        ybords.append(yc + (ycircs[3] - yc) / (xcircs[3] - xc) * (xmin - xc))
        xbords.append(xmin)
        ybords.append(yc + (ycircs[4] - yc) / (xcircs[4] - xc) * (xmin - xc))
        xbords.append(xc + (xcircs[5] - xc) / (ycircs[5] - yc) * (ymin - yc))
        ybords.append(ymin)
        xbords.append(xc + (xcircs[6] - xc) / (ycircs[6] - yc) * (ymin - yc))
        ybords.append(ymin)
        xbords.append(xmax)
        ybords.append(yc + (ycircs[7] - yc) / (xcircs[7] - xc) * (xmax - xc))

        bords = np.vstack([xbords, ybords]).T

        # 3. draw wedges
        draw_wedge(ax, [circs[0], bords[0], (xmax, ymax), bords[1], circs[1]] + cp(angles[0], angles[1], center), color='purple')
        draw_wedge(ax, [circs[1], bords[1], bords[2], circs[2]] + cp(angles[1], angles[2], center), color='deeppink')
        draw_wedge(ax, [circs[2], bords[2], (xmin, ymax), bords[3], circs[3]] + cp(angles[2], angles[3], center), color='dodgerblue')
        draw_wedge(ax, [circs[3], bords[3], bords[4], circs[4]] + cp(angles[3], angles[4], center), color='lawngreen')
        draw_wedge(ax, [circs[4], bords[4], (xmin, ymin), bords[5], circs[5]] + cp(angles[4], angles[5], center), color='orange')
        draw_wedge(ax, [circs[5], bords[5], bords[6], circs[6]] + cp(angles[5], angles[6], center), color='yellow')
        draw_wedge(ax, [circs[6], bords[6], (xmax, ymin), bords[7], circs[7]] + cp(angles[6], angles[7], center), color='tomato')
        draw_wedge(ax, [circs[7], bords[7], bords[0], circs[0]] + cp(angles[7], angles[0], center), color='grey')


    draw_8_slices(ax, (0.1, 0.2), radius=0.7)
    fig.tight_layout()


    plt.ion()
    plt.show()
