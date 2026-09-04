<<<<<<< ours
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

def gui_lay3d(data, fig=None, gs=None):
    shape = data.shape

    state = {
        'glob_x': shape[2] // 2,
        'glob_y': shape[1] // 2,
        'glob_z': shape[0] // 2,
        'mode': 'X',
        'scroll_dir': 0
    }

    # --- FIGURE / GRIDSPEC ---
    if fig is None:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(20, 20, figure=fig,
                               left=0.05, right=0.95,
                               bottom=0.05, top=0.95)

    # --- AXE IMAGE ---
    # On utilise les lignes 0 à 17 pour l'image
    # On commence à la colonne 2 pour laisser de la place aux boutons X,Y,Z à gauche
    ax = fig.add_subplot(gs[0:17, 2:20]) 
    im = ax.imshow(data[:, :, state['glob_x']], cmap='gray', aspect='auto') #'equal'
    ax.set_anchor('C')

    # --- UI BUTTONS (Sélecteurs de mode) ---
    # Indices de lignes : 5, 6, 7 (valides car < 20)
    ax_X = fig.add_subplot(gs[5, 0:2]); btn_X = Button(ax_X, 'X')
    ax_Y = fig.add_subplot(gs[6, 0:2]); btn_Y = Button(ax_Y, 'Y')
    ax_Z = fig.add_subplot(gs[7, 0:2]); btn_Z = Button(ax_Z, 'Z')

    # --- UI NAVIGATION (Ligne 18) ---
    # L'indice 18 est le maximum raisonnable pour laisser la ligne 19 au texte
    ax_prev = fig.add_subplot(gs[18, 4:6])
    btn_prev = Button(ax_prev, '<<')

    ax_slider = fig.add_subplot(gs[18, 7:17])
    # slider sera créé par create_slider()

    ax_next = fig.add_subplot(gs[18, 18:20])
    btn_next = Button(ax_next, '>>')

    # --- TEXTE D'INFORMATION (Ligne 19) ---
    # L'indice 19 est le dernier index autorisé pour une grille de 20
    ax_text = fig.add_subplot(gs[19, 8:15], frameon=False)
    ax_text.set_xticks([]); ax_text.set_yticks([])
    info_text = ax_text.text(0.5, 0.5, '', ha='center', va='center',
                               fontweight='bold', fontsize=11)

    slider = None

    # --- SLIDER ---
    def create_slider(vmax, vinit):
        nonlocal slider
        ax_slider.clear()
        slider = Slider(ax_slider, '', 0, vmax, valinit=vinit, valfmt='%d')
        slider.valtext.set_visible(False)
        slider.on_changed(on_slider)
        return slider

    # --- UPDATE DISPLAY ---
    def update():
        if slider is None:
            return

        idx = int(slider.val)
        mode = state['mode']

        if mode == 'X':
            img = data[:, :, idx]
        elif mode == 'Y':
            img = data[:, idx, :]
        else:
            img = data[idx, :, :]

        im.set_data(img)
        h, w = img.shape
        im.set_extent((0, w, h, 0))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        ax.set_title(f"Mode {mode}", fontweight='bold') #0.55
        ax.axis("off")
        info_text.set_text(f"Slice {idx}/{int(slider.valmax)}")
        fig.canvas.draw_idle()

    def on_slider(val):
        state[f'glob_{state["mode"].lower()}'] = int(val)
        update()

    def switch_mode(mode, dim):
        state['mode'] = mode
        create_slider(shape[dim] - 1,
                      state[f'glob_{mode.lower()}'])
        update()

    # --- AUTO SCROLL ---
    timer = fig.canvas.new_timer(interval=50)

    def scroll():
        if state['scroll_dir'] == 0:
            return
        v = slider.val + state['scroll_dir']
        if 0 <= v <= slider.valmax:
            slider.set_val(int(v))

    timer.add_callback(scroll)

    def press(event):
        if event.inaxes == ax_prev:
            state['scroll_dir'] = -1
            timer.start()
        elif event.inaxes == ax_next:
            state['scroll_dir'] = 1
            timer.start()

    def release(event):
        state['scroll_dir'] = 0
        timer.stop()

    fig.canvas.mpl_connect('button_press_event', press)
    fig.canvas.mpl_connect('button_release_event', release)

    # --- CALLBACKS ---
    btn_X.on_clicked(lambda e: switch_mode('X', 2))
    btn_Y.on_clicked(lambda e: switch_mode('Y', 1))
    btn_Z.on_clicked(lambda e: switch_mode('Z', 0))

    create_slider(shape[2] - 1, state['glob_x'])
    update()
    return fig, btn_X, btn_Y, btn_Z, slider, btn_prev, btn_next, switch_mode
=======
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

def gui_lay3d(data, fig=None, gs=None):
    shape = data.shape

    state = {
        'glob_x': shape[2] // 2,
        'glob_y': shape[1] // 2,
        'glob_z': shape[0] // 2,
        'mode': 'X',
        'scroll_dir': 0
    }

    # --- FIGURE / GRIDSPEC ---
    if fig is None:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(20, 20, figure=fig,
                               left=0.05, right=0.95,
                               bottom=0.05, top=0.95)

    # --- AXE IMAGE ---
    # On utilise les lignes 0 à 17 pour l'image
    # On commence à la colonne 2 pour laisser de la place aux boutons X,Y,Z à gauche
    ax = fig.add_subplot(gs[0:17, 2:20]) 
    im = ax.imshow(data[:, :, state['glob_x']], cmap='gray', aspect='auto') #'equal'
    ax.set_anchor('C')

    # --- UI BUTTONS (Sélecteurs de mode) ---
    # Indices de lignes : 5, 6, 7 (valides car < 20)
    ax_X = fig.add_subplot(gs[5, 0:2]); btn_X = Button(ax_X, 'X')
    ax_Y = fig.add_subplot(gs[6, 0:2]); btn_Y = Button(ax_Y, 'Y')
    ax_Z = fig.add_subplot(gs[7, 0:2]); btn_Z = Button(ax_Z, 'Z')

    # --- UI NAVIGATION (Ligne 18) ---
    # L'indice 18 est le maximum raisonnable pour laisser la ligne 19 au texte
    ax_prev = fig.add_subplot(gs[18, 4:6])
    btn_prev = Button(ax_prev, '<<')

    ax_slider = fig.add_subplot(gs[18, 7:17])
    # slider sera créé par create_slider()

    ax_next = fig.add_subplot(gs[18, 18:20])
    btn_next = Button(ax_next, '>>')

    # --- TEXTE D'INFORMATION (Ligne 19) ---
    # L'indice 19 est le dernier index autorisé pour une grille de 20
    ax_text = fig.add_subplot(gs[19, 8:15], frameon=False)
    ax_text.set_xticks([]); ax_text.set_yticks([])
    info_text = ax_text.text(0.5, 0.5, '', ha='center', va='center',
                               fontweight='bold', fontsize=11)

    slider = None

    # --- SLIDER ---
    def create_slider(vmax, vinit):
        nonlocal slider
        ax_slider.clear()
        slider = Slider(ax_slider, '', 0, vmax, valinit=vinit, valfmt='%d')
        slider.valtext.set_visible(False)
        slider.on_changed(on_slider)
        return slider

    # --- UPDATE DISPLAY ---
    def update():
        if slider is None:
            return

        idx = int(slider.val)
        mode = state['mode']

        if mode == 'X':
            img = data[:, :, idx]
        elif mode == 'Y':
            img = data[:, idx, :]
        else:
            img = data[idx, :, :]

        im.set_data(img)
        h, w = img.shape
        im.set_extent((0, w, h, 0))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        ax.set_title(f"Mode {mode}", fontweight='bold') #0.55
        ax.axis("off")
        info_text.set_text(f"Slice {idx}/{int(slider.valmax)}")
        fig.canvas.draw_idle()

    def on_slider(val):
        state[f'glob_{state["mode"].lower()}'] = int(val)
        update()

    def switch_mode(mode, dim):
        state['mode'] = mode
        create_slider(shape[dim] - 1,
                      state[f'glob_{mode.lower()}'])
        update()

    # --- AUTO SCROLL ---
    timer = fig.canvas.new_timer(interval=50)

    def scroll():
        if state['scroll_dir'] == 0:
            return
        v = slider.val + state['scroll_dir']
        if 0 <= v <= slider.valmax:
            slider.set_val(int(v))

    timer.add_callback(scroll)

    def press(event):
        if event.inaxes == ax_prev:
            state['scroll_dir'] = -1
            timer.start()
        elif event.inaxes == ax_next:
            state['scroll_dir'] = 1
            timer.start()

    def release(event):
        state['scroll_dir'] = 0
        timer.stop()

    fig.canvas.mpl_connect('button_press_event', press)
    fig.canvas.mpl_connect('button_release_event', release)

    # --- CALLBACKS ---
    btn_X.on_clicked(lambda e: switch_mode('X', 2))
    btn_Y.on_clicked(lambda e: switch_mode('Y', 1))
    btn_Z.on_clicked(lambda e: switch_mode('Z', 0))

    create_slider(shape[2] - 1, state['glob_x'])
    update()
    return fig, btn_X, btn_Y, btn_Z, slider, btn_prev, btn_next, switch_mode
>>>>>>> theirs
