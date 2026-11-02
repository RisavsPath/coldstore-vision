"""ColdStore-Vision explainer: procedural cold-store + inspection drone scene (Blender 4.1, EEVEE).

Run inside Blender (MCP or Text Editor):  exec(open(r"<path>/build_scene.py").read())
or headless:  blender -b -P build_scene.py -- --save scene.blend
Everything is generated here (no external models) so the scene is fully reproducible.

Layout (metres): aisle along +X, racks at y = +/-RACK_Y, dock chamber at x = -6.
Timeline: 30 fps, frames 1..FRAME_END (see ANIMATION section).
"""
import math
import random

import bpy
import bmesh
from mathutils import Vector

FPS = 30
FRAME_END = 900
BAYS = 5
BAY_LEN = 2.7
LEVELS = (0.15, 1.55, 2.95)          # beam heights (top of each pallet level)
RACK_Y = 2.6                          # rack row centre distance from aisle centre
RACK_DEPTH = 1.1
STATUS = {"ok": (0.18, 0.62, 0.31), "warn": (0.88, 0.66, 0.0), "crit": (0.84, 0.23, 0.23),
          "miss": (0.08, 0.72, 0.78), "unv": (0.54, 0.56, 0.6)}
rng = random.Random(7)


# ----------------------------------------------------------------------------- helpers
def clear_scene():
    # remove directly: objects hidden by visibility keyframes can't be selected, so
    # select_all + delete would leave them behind and duplicates would pile up on rebuild
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for a in list(bpy.data.actions):  # stale actions from earlier builds
        bpy.data.actions.remove(a)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    bpy.context.scene.timeline_markers.clear()
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.curves,
                 bpy.data.images, bpy.data.node_groups):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def coll(name):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(c)
    return c


def link(obj, collection):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def bsdf(mat):
    return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def mat_basic(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0, alpha=1.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = bsdf(m)
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
        m.shadow_method = "NONE"
    return m


def box(name, size, loc, mat=None, collection=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    if mat:
        o.data.materials.append(mat)
    if collection:
        link(o, collection)
    return o


# ----------------------------------------------------------------------------- materials
def sack_material(name, base, mesh_color, scale=60.0):
    """Woven mesh sack: base colour + fine weave (wave texture) bump + slight colour noise."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type not in ("BSDF_PRINCIPLED", "OUTPUT_MATERIAL"):
            nt.nodes.remove(n)
    b = bsdf(m)
    tc = nt.nodes.new("ShaderNodeTexCoord")  # object coords; distortion breaks the moire on curved sacks
    w1 = nt.nodes.new("ShaderNodeTexWave")
    w1.inputs["Scale"].default_value = scale
    w1.bands_direction = "DIAGONAL"
    w1.inputs["Distortion"].default_value = 3.0
    w2 = nt.nodes.new("ShaderNodeTexWave")
    w2.inputs["Scale"].default_value = scale
    w2.bands_direction = "X"
    w2.inputs["Distortion"].default_value = 3.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY"
    ramp = nt.nodes.new("ShaderNodeMix")
    ramp.data_type = "RGBA"
    ramp.inputs["A"].default_value = (*base, 1)
    ramp.inputs["B"].default_value = (*mesh_color, 1)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.45
    nt.links.new(tc.outputs["Object"], w1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], w2.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(w1.outputs["Fac"], mix.inputs[0])
    nt.links.new(w2.outputs["Fac"], mix.inputs[1])
    nt.links.new(mix.outputs[0], ramp.inputs["Factor"])
    nt.links.new(ramp.outputs["Result"], b.inputs["Base Color"])
    nt.links.new(mix.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = 0.85
    return m


# ----------------------------------------------------------------------------- environment
def build_world():
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAME_END
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    try:
        sc.render.engine = "BLENDER_EEVEE"
    except TypeError:
        pass
    ee = sc.eevee
    for attr, val in (("use_bloom", True), ("bloom_intensity", 0.06), ("use_gtao", True), ("use_ssr", True),
                      ("use_soft_shadows", True), ("volumetric_start", 0.5), ("volumetric_end", 60.0),
                      ("volumetric_tile_size", "8"), ("taa_render_samples", 64)):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, val)
            except (TypeError, ValueError):
                pass
    sc.view_settings.view_transform = "Filmic" if "Filmic" in [i.identifier for i in
        bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items] else sc.view_settings.view_transform
    sc.view_settings.look = "None"
    sc.view_settings.exposure = -0.4
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.012, 0.02, 0.035, 1)
    bg.inputs["Strength"].default_value = 1.0
    # cold-store mist: low-density volume scatter in the world
    out = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
    vol = nt.nodes.get("Mist") or nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.name = "Mist"
    vol.inputs["Density"].default_value = 0.008
    vol.inputs["Color"].default_value = (0.75, 0.86, 1.0, 1)
    nt.links.new(vol.outputs[0], out.inputs["Volume"])


def build_room():
    c = coll("Room")
    floor_mat = bpy.data.materials.get("hangar_concrete_floor") or mat_basic("Floor", (0.2, 0.21, 0.22), rough=0.45)
    wall_mat = mat_basic("InsulatedPanel", (0.55, 0.6, 0.66), rough=0.6)
    length, width, height = 26.0, 11.0, 5.2
    x0 = -9.0
    box("Floor", (length, width, 0.1), (x0 + length / 2, 0, -0.05), floor_mat, c)
    box("Ceiling", (length, width, 0.1), (x0 + length / 2, 0, height), wall_mat, c)
    box("WallBack", (length, 0.15, height), (x0 + length / 2, width / 2, height / 2), wall_mat, c)
    box("WallFront", (length, 0.15, height), (x0 + length / 2, -width / 2, height / 2), wall_mat, c)
    box("WallEnd", (0.15, width, height), (x0 + length, 0, height / 2), wall_mat, c)
    box("WallStart", (0.15, width, height), (x0, 0, height / 2), wall_mat, c)
    # panel seams on the back wall for scale
    seam = mat_basic("Seam", (0.55, 0.6, 0.66), rough=0.6)
    for i in range(int(length / 1.2)):
        box(f"Seam{i}", (0.03, 0.02, height), (x0 + 0.6 + i * 1.2, width / 2 - 0.08, height / 2), seam, c)
    # yellow aisle floor lines
    line = mat_basic("AisleLine", (0.95, 0.75, 0.05), rough=0.4)
    for y in (-1.45, 1.45):
        box(f"AisleLine{y}", (length - 1, 0.08, 0.01), (x0 + length / 2, y, 0.005), line, c)
    # ceiling light strips (cool white)
    strip = mat_basic("LightStrip", (1, 1, 1), emit=(0.85, 0.93, 1.0), emit_strength=12.0)
    for i in range(6):
        x = x0 + 2.5 + i * 4.0
        box(f"Strip{i}", (2.4, 0.18, 0.05), (x, 0, height - 0.08), strip, c)
        bpy.ops.object.light_add(type="AREA", location=(x, 0, height - 0.2))
        L = bpy.context.active_object
        L.name = f"StripLight{i}"
        L.data.shape = "RECTANGLE"
        L.data.size, L.data.size_y = 2.4, 0.4
        L.data.energy = 260
        L.data.color = (0.86, 0.93, 1.0)
        link(L, c)


# ----------------------------------------------------------------------------- racks + sacks
def sack_mesh():
    """One rounded, slightly bulging sack mesh shared by every sack instance."""
    me = bpy.data.meshes.get("SackMesh")
    if me:
        return me
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=4, use_grid_fill=True)
    for v in bm.verts:
        co = v.co.copy()
        # superellipse-ish bulge: round the corners, fatten the middle
        r = co.length
        v.co = co.normalized() * (0.55 * r + 0.32)
        v.co.x *= 0.62
        v.co.y *= 0.42
        v.co.z *= 0.36
        v.co.z += 0.04 * (1 - abs(co.x) * 2)  # sag in the middle
    me = bpy.data.meshes.new("SackMesh")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(None)  # one slot; each sack object links its own material into it
    for p in me.polygons:
        p.use_smooth = True
    return me


def qr_image(name, seed, n=21):
    img = bpy.data.images.get(name)
    if img:
        return img
    r = random.Random(seed)
    px = 8
    size = n * px
    img = bpy.data.images.new(name, size, size)
    data = [1.0] * (size * size * 4)

    def cell(cx, cy, on):
        v = 0.0 if on else 1.0
        for yy in range(cy * px, cy * px + px):
            for xx in range(cx * px, cx * px + px):
                i = (yy * size + xx) * 4
                data[i] = data[i + 1] = data[i + 2] = v

    finder = lambda x, y: (x in (0, 6) or y in (0, 6) or (2 <= x <= 4 and 2 <= y <= 4))  # noqa: E731
    for cy in range(n):
        for cx in range(n):
            for ox, oy in ((0, 0), (n - 7, 0), (0, n - 7)):
                if ox <= cx < ox + 7 and oy <= cy < oy + 7:
                    cell(cx, cy, finder(cx - ox, cy - oy))
                    break
            else:
                cell(cx, cy, r.random() < 0.5)
    img.pixels = data
    img.pack()
    return img


def tag_material(name, seed):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = bsdf(m)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = qr_image(f"QR_{name}", seed)
    tex.interpolation = "Closest"
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.3
    b.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    b.inputs["Emission Strength"].default_value = 0.15
    return m


def build_racks():
    c = coll("Racks")
    cs = coll("Sacks")
    upright = mat_basic("RackUpright", (0.06, 0.22, 0.55), rough=0.4, metal=0.6)
    beam = mat_basic("RackBeam", (0.95, 0.42, 0.05), rough=0.4, metal=0.5)
    pallet = mat_basic("Pallet", (0.55, 0.4, 0.24), rough=0.8)
    onion = sack_material("SackOnion", (0.22, 0.015, 0.035), (0.55, 0.07, 0.12))
    potato = sack_material("SackPotato", (0.2, 0.12, 0.055), (0.42, 0.29, 0.15))
    me = sack_mesh()
    x_start = 0.0
    sacks = []
    for side, sy in (("L", RACK_Y), ("R", -RACK_Y)):
        for b in range(BAYS + 1):
            x = x_start + b * BAY_LEN
            for dy in (-RACK_DEPTH / 2, RACK_DEPTH / 2):
                box(f"Upright_{side}{b}_{dy:+.1f}", (0.09, 0.09, 4.2), (x, sy + dy, 2.1), upright, c)
        for b in range(BAYS):
            xc = x_start + (b + 0.5) * BAY_LEN
            for li, z in enumerate(LEVELS):
                for dy in (-RACK_DEPTH / 2, RACK_DEPTH / 2):
                    box(f"Beam_{side}{b}_{li}_{dy:+.1f}", (BAY_LEN - 0.1, 0.07, 0.14), (xc, sy + dy, z - 0.07), beam, c)
                box(f"Pallet_{side}{b}_{li}", (BAY_LEN - 0.35, RACK_DEPTH - 0.1, 0.12), (xc, sy, z + 0.06), pallet, c)
                mat = onion if (b + li + (side == "R")) % 2 == 0 else potato
                # 4 along x, 2 deep, 2 high
                for k in range(4):
                    for d in range(2):
                        for h in range(2):
                            if li == 2 and h == 1:
                                continue
                            sx = xc + (k - 1.5) * 0.6 + rng.uniform(-0.03, 0.03)
                            syy = sy + (d - 0.5) * 0.48 * (1 if side == "L" else -1)
                            sz = z + 0.12 + 0.2 + h * 0.37
                            o = bpy.data.objects.new(f"Sack_{side}{b}_{li}_{k}{d}{h}", me)
                            o.location = (sx, syy, sz)
                            o.rotation_euler = (rng.uniform(-.05, .05), rng.uniform(-.05, .05), rng.uniform(-.08, .08))
                            cs.objects.link(o)
                            # mesh is shared by every sack: material lives on the object slot
                            o.material_slots[0].link = "OBJECT"
                            o.material_slots[0].material = mat
                            sacks.append(o)
        # bay tags (QR) on the aisle-facing upright, one per bay
        for b in range(BAYS):
            x = x_start + b * BAY_LEN + 0.25
            face_y = sy - (RACK_DEPTH / 2 + 0.06) * (1 if side == "L" else -1)
            t = box(f"Tag_{side}{b}", (0.32, 0.01, 0.32), (x, face_y, 1.2), tag_material(f"Tag_{side}{b}", hash((side, b)) % 999), c)
            if side == "R":
                t.rotation_euler.z = math.pi
            lab = bpy.data.curves.new(f"TagLabel_{side}{b}", "FONT")
            lab.body = f"{'A' if side == 'L' else 'B'}-{b + 1:02d}"
            lab.size = 0.11
            lab.align_x = "CENTER"
            lo = bpy.data.objects.new(f"TagLabel_{side}{b}", lab)
            lo.location = (x, face_y - 0.006 * (1 if side == "L" else -1), 0.98)
            lo.rotation_euler = (math.pi / 2, 0, 0 if side == "L" else math.pi)
            lo.data.materials.append(mat_basic("TagText", (1, 1, 1), emit=(1, 1, 1), emit_strength=1.5))
            c.objects.link(lo)
    return sacks


# ----------------------------------------------------------------------------- timeline (frames @30fps)
T_DOOR = (20, 70)          # dock door opens
T_EXIT = (70, 150)         # drone leaves the chamber
T_FLY = (150, 300)         # flight down the aisle
T_TAG = (300, 420)         # hover + read bay QR tag
T_PASS = (430, 700)        # inspection pass along the rack
T_END = 900
HOVER_DX = 0.75            # hover this far before the tag so it stays visible
HOVER_Z = 1.3
DOCK_X = -6.6
SCAN_BAYS = (2, 3, 4)      # bays inspected on the A (left) side
PASS_Y = 0.95              # drone distance from aisle centre during the pass (towards rack A)
PASS_X = (SCAN_BAYS[0] * BAY_LEN - 0.2, (SCAN_BAYS[-1] + 1) * BAY_LEN + 0.2)


def key(obj, frame, loc=None, rot=None, scale=None, interp="BEZIER"):
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert("location", frame=frame)
    if rot is not None:
        obj.rotation_euler = rot
        obj.keyframe_insert("rotation_euler", frame=frame)
    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert("scale", frame=frame)
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                if kp.co.x == frame:
                    kp.interpolation = interp


def show_between(obj, f_on, f_off=None):
    """Appear at f_on (and vanish at f_off) by keying SCALE with constant interpolation.

    Keyframing hide_viewport/hide_render is unreliable: the depsgraph can skip re-evaluating such
    objects, leaving them stuck at their creation transform (e.g. at the origin). Scale is always
    evaluated, so viewport previews and final renders agree.
    """
    full = obj.scale.copy()
    obj.scale = (0, 0, 0)
    obj.keyframe_insert("scale", frame=max(1, f_on - 1))
    obj.scale = full
    obj.keyframe_insert("scale", frame=f_on)
    if f_off:
        obj.scale = (0, 0, 0)
        obj.keyframe_insert("scale", frame=f_off)
    obj.scale = full
    for fc in obj.animation_data.action.fcurves:
        if fc.data_path == "scale":
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def emit_mat(name, color, strength=4.0, alpha=1.0):
    return mat_basic(name, color, rough=0.4, emit=color, emit_strength=strength, alpha=alpha)


def text_obj(name, body, size, mat, collection, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    o = bpy.data.objects.new(name, cu)
    o.data.materials.append(mat)
    collection.objects.link(o)
    return o


def cyl(name, r, depth, loc, mat, collection, rot=(0, 0, 0), verts=32):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, rotation=rot, vertices=verts)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return link(o, collection)


# ----------------------------------------------------------------------------- dock chamber
def build_dock():
    c = coll("Dock")
    shell = mat_basic("DockShell", (0.12, 0.14, 0.17), rough=0.35, metal=0.7)
    trim = emit_mat("DockTrim", (0.1, 0.65, 1.0), 6.0)
    inner = emit_mat("DockInner", (0.75, 0.88, 1.0), 2.5)
    x = DOCK_X
    box("DockBase", (1.6, 1.6, 0.08), (x, 0, 0.42), shell, c)
    for sy in (-0.78, 0.78):
        box(f"DockSide{sy}", (1.6, 0.06, 1.25), (x, sy, 1.0), shell, c)
    box("DockRoof", (1.6, 1.6, 0.08), (x, 0, 1.62), shell, c)
    box("DockBackWall", (0.06, 1.6, 1.25), (x - 0.78, 0, 1.0), inner, c)
    box("DockPad", (0.9, 0.9, 0.02), (x, 0, 0.47), emit_mat("DockPadGlow", (0.1, 0.65, 1.0), 1.5), c)
    for i, yy in enumerate((-0.8, 0.8)):
        box(f"DockTrim{i}", (0.04, 0.04, 1.3), (x + 0.8, yy, 1.0), trim, c)
    for i in range(4):
        box(f"DockLeg{i}", (0.08, 0.08, 0.4), (x + (-0.7 if i < 2 else 0.7), (-0.7 if i % 2 else 0.7), 0.2), shell, c)
    door = box("DockDoor", (0.05, 1.5, 1.2), (x + 0.79, 0, 1.0), shell, c)
    key(door, T_DOOR[0], loc=(x + 0.79, 0, 1.0))
    key(door, T_DOOR[1], loc=(x + 0.79, 0, 2.25))
    lo = text_obj("DockLabel", "DRONE DOCK  ·  CHARGING", 0.09, emit_mat("DockLabelMat", (0.6, 0.85, 1.0), 3.0), c)
    lo.location = (x + 0.83, 0, 1.68)
    lo.rotation_euler = (math.pi / 2, 0, math.pi / 2)


# ----------------------------------------------------------------------------- drone
def build_drone():
    c = coll("Drone")
    body_m = mat_basic("DroneBody", (0.05, 0.055, 0.065), rough=0.3, metal=0.4)
    arm_m = mat_basic("DroneArm", (0.12, 0.13, 0.15), rough=0.4, metal=0.6)
    prop_m = mat_basic("DroneProp", (0.08, 0.09, 0.1), rough=0.3, alpha=0.16)
    lens_m = mat_basic("Lens", (0.01, 0.01, 0.02), rough=0.05, metal=0.2)
    cyan = emit_mat("LedCyan", (0.1, 0.8, 1.0), 12)
    green = emit_mat("LedGreen", (0.2, 1.0, 0.4), 10)
    red = emit_mat("LedRed", (1.0, 0.1, 0.1), 10)

    root = bpy.data.objects.new("Drone", None)
    root.empty_display_size = 0.3
    c.objects.link(root)
    parts = [box("DroneBody", (0.36, 0.24, 0.09), (0, 0, 0), body_m, c, bevel=0.025),
             box("DroneShell", (0.26, 0.18, 0.05), (0, 0, 0.065), arm_m, c, bevel=0.02)]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        mx, my = 0.27 * sx, 0.27 * sy
        parts.append(cyl(f"DroneArm{i}", 0.013, 0.36, (mx / 2, my / 2, 0.01), arm_m, c,
                         rot=(0, math.pi / 2, math.atan2(my, mx))))
        parts.append(cyl(f"DroneMotor{i}", 0.03, 0.06, (mx, my, 0.02), body_m, c))
        parts.append(cyl(f"DronePropDisc{i}", 0.15, 0.004, (mx, my, 0.06), prop_m, c, verts=48))
        blade = box(f"DroneBlade{i}", (0.29, 0.025, 0.006), (mx, my, 0.062), arm_m, c)
        drv = blade.driver_add("rotation_euler", 2).driver
        drv.expression = f"frame*{1.7 if i % 2 else -1.7}"
        parts.append(blade)
        bpy.ops.curve.primitive_bezier_circle_add(radius=0.16, location=(mx, my, 0.06))
        g = bpy.context.active_object
        g.name = f"DroneGuard{i}"
        g.data.bevel_depth = 0.006
        g.data.materials.append(arm_m)
        parts.append(link(g, c))
    # sensors: front gimbal camera, top LiDAR puck, VIO stereo pair, edge computer
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.045, location=(0.2, 0, -0.06))
    gimbal = bpy.context.active_object
    gimbal.name = "DroneGimbal"
    gimbal.data.materials.append(arm_m)
    parts.append(link(gimbal, c))
    parts.append(cyl("DroneCamLens", 0.022, 0.03, (0.243, 0, -0.06), lens_m, c, rot=(0, math.pi / 2, 0)))
    parts.append(cyl("DroneLidar", 0.055, 0.05, (0, 0, 0.115), body_m, c))
    bpy.ops.mesh.primitive_torus_add(major_radius=0.056, minor_radius=0.006, location=(0, 0, 0.12))
    ring = bpy.context.active_object
    ring.name = "DroneLidarRing"
    ring.data.materials.append(cyan)
    parts.append(link(ring, c))
    for i, yy in enumerate((-0.05, 0.05)):
        parts.append(cyl(f"DroneVIO{i}", 0.012, 0.02, (0.181, yy, 0.02), lens_m, c, rot=(0, math.pi / 2, 0)))
    parts.append(box("DroneEdgeAI", (0.12, 0.1, 0.02), (-0.06, 0, 0.1), arm_m, c))
    parts.append(box("DroneLedFront", (0.01, 0.12, 0.012), (0.181, 0, -0.025), green, c))
    parts.append(box("DroneLedRear", (0.01, 0.12, 0.012), (-0.181, 0, -0.025), red, c))
    for p in parts:
        p.parent = root
    bpy.ops.object.light_add(type="POINT", location=(0.3, 0, -0.1))
    gl = bpy.context.active_object
    gl.name = "DroneGlow"
    gl.data.energy = 6
    gl.data.color = (0.6, 0.85, 1.0)
    gl.parent = root
    link(gl, c)
    return root


def build_callouts(drone):
    """Labels for the onboard devices; they ride with the drone and face the chase camera."""
    c = coll("Callouts")
    items = [("4K camera  ·  gimbal", (0.25, 0.0, -0.07), (0.5, -0.2, -0.2)),
             ("LiDAR  ·  depth sensing", (0.0, 0.0, 0.13), (-0.15, -0.2, 0.3)),
             ("VIO tracking  ·  6-DoF pose", (0.18, 0.05, 0.02), (0.6, -0.2, 0.18)),
             ("Edge AI  ·  YOLO11-seg", (-0.06, 0.0, 0.11), (-0.62, -0.2, 0.1))]
    line_m = emit_mat("CalloutLine", (0.3, 0.75, 1.0), 6)
    text_m = emit_mat("CalloutText", (0.92, 0.96, 1.0), 4)
    card_m = emit_mat("CalloutCard", (0.02, 0.06, 0.14), 1.0, alpha=0.85)
    for i, (txt, anchor, off) in enumerate(items):
        end = (anchor[0] + off[0], anchor[1] + off[1], anchor[2] + off[2])
        t = text_obj(f"CalloutTxt{i}", txt, 0.07, text_m, c)
        t.data.align_y = "CENTER"
        t.parent = drone
        t.location = (end[0], end[1], end[2] + 0.045)
        con = t.constraints.new("TRACK_TO")
        con.track_axis, con.up_axis = "TRACK_Z", "UP_Y"
        # dark card behind the text so it reads against the bright insulated walls
        w = len(txt) * 0.07 * 0.52 + 0.08
        bpy.ops.mesh.primitive_plane_add(size=1)
        card = bpy.context.active_object
        card.name = f"CalloutCard{i}"
        card.scale = (w, 0.11, 1)
        card.location = (0, 0, -0.004)
        card.data.materials.append(card_m)
        link(card, c)
        card.parent = t
        show_between(card, T_EXIT[1] + 5 + i * 12, T_FLY[1] - 40)
        crv = bpy.data.curves.new(f"CalloutLine{i}", "CURVE")
        crv.dimensions = "3D"
        sp = crv.splines.new("POLY")
        sp.points.add(1)
        sp.points[0].co = (*anchor, 1)
        sp.points[1].co = (*end, 1)
        crv.bevel_depth = 0.004
        ln = bpy.data.objects.new(f"CalloutLine{i}", crv)
        ln.data.materials.append(line_m)
        c.objects.link(ln)
        ln.parent = drone
        start = T_EXIT[1] + 5 + i * 12
        show_between(t, start, T_FLY[1] - 40)
        show_between(ln, start, T_FLY[1] - 40)


# ----------------------------------------------------------------------------- scan effects + statuses
def build_scan_fx(drone, sacks):
    c = coll("ScanFX")
    tag_x = SCAN_BAYS[0] * BAY_LEN + 0.25
    face_y = RACK_Y - (RACK_DEPTH / 2 + 0.06)
    # QR-read cone from the drone camera to the tag
    bpy.ops.mesh.primitive_cone_add(radius1=0.22, radius2=0.01, depth=1.0, vertices=32)
    cone = bpy.context.active_object
    cone.name = "TagScanCone"
    cone.data.materials.append(emit_mat("ScanBeam", (0.2, 0.85, 1.0), 4.0, alpha=0.32))
    link(cone, c)
    src = Vector((tag_x - HOVER_DX + 0.0, PASS_Y + 0.25, HOVER_Z - 0.06))   # drone camera lens (yawed +90deg)
    dst = Vector((tag_x, face_y - 0.02, 1.2))
    d = src - dst
    cone.location = (src + dst) / 2
    cone.rotation_euler = d.to_track_quat("Z", "Y").to_euler()   # tip (+Z) at the drone, base on the tag
    cone.scale = (1, 1, d.length)
    show_between(cone, T_TAG[0] + 20, T_TAG[1] - 10)
    hl = box("TagHighlight", (0.4, 0.012, 0.4), (tag_x, face_y - 0.012, 1.2), emit_mat("TagGlow", (0.2, 1.0, 0.5), 5, 0.5), c)
    show_between(hl, T_TAG[0] + 45, T_TAG[1] - 10)
    ok = text_obj("TagReadTxt", "RACK A-03  ✓  ID READ", 0.09, emit_mat("TagReadMat", (0.3, 1.0, 0.55), 5), c)
    ok.location = (tag_x, face_y - 0.03, 1.47)
    ok.rotation_euler = (math.pi / 2, 0, 0)
    show_between(ok, T_TAG[0] + 50, T_TAG[1] - 10)
    # vertical laser sheet that travels with the drone during the pass
    sheet_h = 3.3
    sheet = box("LaserSheet", (0.012, abs(face_y - PASS_Y), sheet_h), (0, 0, 0),
                emit_mat("Laser", (0.1, 0.9, 1.0), 3, 0.14), c)
    sheet.parent = drone
    # local frame: drone yawed +90deg, so local -X points at rack A
    sheet.location = (-abs(face_y - PASS_Y) / 2, 0, 1.75 - 1.8)
    sheet.rotation_euler = (0, 0, math.pi / 2)
    show_between(sheet, T_PASS[0], T_PASS[1])

    # aisle-facing sacks in the scanned bays get a status outline when the sheet passes them
    scanned = [s for s in sacks if s.name.startswith("Sack_L") and int(s.name.split("_")[1][1]) in SCAN_BAYS
               and s.name.split("_")[-1][1] == "0"]
    rnd = random.Random(3)
    picks = rnd.sample(scanned, 4)
    special = {picks[0].name: "crit", picks[1].name: "miss", picks[2].name: "warn", picks[3].name: "warn"}
    outline_mats = {k: emit_mat(f"Status_{k}", v, 2.2, 0.95) for k, v in STATUS.items()}
    span = PASS_X[1] - PASS_X[0]
    for s in scanned:
        st = special.get(s.name, "ok")
        frac = (s.location.x - PASS_X[0]) / span
        f_on = int(T_PASS[0] + frac * (T_PASS[1] - T_PASS[0])) + 4
        o = bpy.data.objects.new(f"Outline_{s.name}", s.data)
        o.location, o.rotation_euler = s.location.copy(), s.rotation_euler.copy()
        o.scale = (1.06, 1.06, 1.06)
        c.objects.link(o)
        o.material_slots[0].link = "OBJECT"
        o.material_slots[0].material = outline_mats[st]
        o.modifiers.new("wire", "WIREFRAME").thickness = 0.008
        show_between(o, f_on)
        if st == "miss":  # present on the previous scan, gone today
            s.hide_render = s.hide_viewport = True
        if st == "crit":
            s.material_slots[0].material = mat_basic("SackRot", (0.07, 0.045, 0.03), rough=0.9)
    return special


def build_panel():
    """Holographic summary panel for the closing shot."""
    c = coll("Panel")
    x = (SCAN_BAYS[0] + SCAN_BAYS[-1] + 1) / 2 * BAY_LEN
    y, z = 1.0, 3.25
    objs = [box("HoloPanel", (2.7, 0.02, 0.78), (x, y, z), emit_mat("HoloBG", (0.02, 0.1, 0.22), 1.2, 0.75), c)]
    rows = [("BAYS A-03 → A-05  ·  vs scan of 21 Oct", 0.095, (0.85, 0.93, 1.0), -1.25, 0.2),
            ("●  NORMAL  44", 0.11, STATUS["ok"], -1.25, 0.0), ("●  WARNING  2", 0.11, STATUS["warn"], -1.25, -0.2),
            ("●  CRITICAL  1  · spoilage", 0.11, STATUS["crit"], 0.1, 0.0), ("●  MISSING  1", 0.11, STATUS["miss"], 0.1, -0.2)]
    for i, (txt, size, col, dx, dz) in enumerate(rows):
        t = text_obj(f"PanelTxt{i}", txt, size, emit_mat(f"PanelTxtMat{i}", col, 5), c, align="LEFT")
        t.location = (x + dx, y - 0.03, z + dz)
        t.rotation_euler = (math.pi / 2, 0, 0)
        objs.append(t)
    for o in objs:
        show_between(o, T_PASS[1] + 25)


# ----------------------------------------------------------------------------- animation + cameras
def animate_drone(drone):
    tag_x = SCAN_BAYS[0] * BAY_LEN + 0.25
    face_left = (0, 0, math.pi / 2)
    key(drone, 1, loc=(DOCK_X, 0, 0.6), rot=(0, 0, 0))
    key(drone, T_EXIT[0], loc=(DOCK_X, 0, 0.62), rot=(0, 0, 0))
    key(drone, T_EXIT[0] + 30, loc=(DOCK_X + 0.3, 0, 1.05))
    key(drone, T_EXIT[1], loc=(DOCK_X + 2.2, 0, 1.5), rot=(0, 0.12, 0))
    key(drone, T_FLY[0] + 20, loc=(DOCK_X + 3.2, 0, 1.5), rot=(0, 0.1, 0))
    key(drone, T_FLY[1] - 30, loc=(tag_x - 1.2, 0.2, 1.35), rot=(0, 0.12, 0))
    key(drone, T_FLY[1], loc=(tag_x - HOVER_DX, PASS_Y - 0.1, HOVER_Z), rot=face_left)
    key(drone, T_TAG[0], loc=(tag_x - HOVER_DX, PASS_Y, HOVER_Z), rot=face_left)
    key(drone, T_TAG[1], loc=(tag_x - HOVER_DX, PASS_Y, HOVER_Z), rot=face_left)
    key(drone, T_PASS[0], loc=(PASS_X[0], PASS_Y, 1.8), rot=face_left)
    key(drone, T_PASS[1], loc=(PASS_X[1], PASS_Y, 1.8), rot=face_left)
    key(drone, T_END, loc=(PASS_X[1] + 0.6, PASS_Y - 0.4, 2.3), rot=face_left)
    for fc in drone.animation_data.action.fcurves:  # constant-speed inspection pass
        for kp in fc.keyframe_points:
            if T_PASS[0] <= kp.co.x < T_PASS[1]:
                kp.interpolation = "LINEAR"
    m = drone.animation_data.action.fcurves.find("location", index=2).modifiers.new("NOISE")
    m.scale, m.strength = 25, 0.03  # hover bob


def camera(name, lens=28):
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    cam.data.lens = lens
    coll("Cameras").objects.link(cam)
    return cam


def aim(cam, target):
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"


def empty(name, loc, parent=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    coll("Cameras").objects.link(e)
    if parent:
        e.parent = parent
    return e


def build_cameras(drone):
    sc = bpy.context.scene
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    tag_x = SCAN_BAYS[0] * BAY_LEN + 0.25
    mid = (SCAN_BAYS[0] + SCAN_BAYS[-1] + 1) / 2 * BAY_LEN
    c1 = camera("Cam1_Dock", 32)            # dock reveal
    key(c1, 1, loc=(DOCK_X + 3.4, -2.4, 1.3))
    key(c1, T_EXIT[1], loc=(DOCK_X + 3.0, -1.9, 1.5))
    aim(c1, drone)
    rig = empty("ChaseRig", (0, 0, 0))     # follows the drone's position only (not its tilt)
    cl = rig.constraints.new("COPY_LOCATION")
    cl.target = drone
    c2 = camera("Cam2_Chase", 22)           # chase cam with callouts; stays inside the aisle (rack B face at y=-2.05)
    c2.parent = rig
    key(c2, T_EXIT[1] - 10, loc=(1.5, -1.5, 0.35))
    key(c2, T_FLY[1] - 40, loc=(1.2, -1.4, 0.3))
    aim(c2, drone)
    c3 = camera("Cam3_Tag", 30)             # over-the-shoulder tag scan
    key(c3, T_FLY[1] - 30, loc=(tag_x + 1.6, -0.9, 1.7))   # from the far side: drone left of the tag
    key(c3, T_TAG[1], loc=(tag_x + 1.1, -0.5, 1.5))
    aim(c3, empty("Cam3Target", (tag_x - 0.35, 1.6, 1.25)))
    c4 = camera("Cam4_Pass", 26)            # side tracking during the pass
    key(c4, T_PASS[0] - 10, loc=(PASS_X[0] - 1.4, -1.3, 2.0))
    key(c4, T_PASS[1], loc=(PASS_X[1] - 1.4, -1.3, 2.0), interp="LINEAR")
    aim(c4, empty("Cam4Target", (0.0, 1.1, -0.2), parent=drone))
    c5 = camera("Cam5_Reveal", 20)          # wide reveal + panel
    key(c5, T_PASS[1], loc=(mid - 1.0, -1.6, 1.8))
    key(c5, T_END, loc=(mid, -1.9, 2.4))
    aim(c5, empty("Cam5Target", (mid, RACK_Y - 0.4, 2.15)))
    for f, cam in ((1, c1), (T_EXIT[1] - 10, c2), (T_FLY[1] - 30, c3), (T_PASS[0] - 10, c4), (T_PASS[1] + 10, c5)):
        sc.timeline_markers.new(cam.name, frame=f).camera = cam
    sc.camera = c1
    for o in bpy.data.objects:
        if o.name.startswith("CalloutTxt"):
            o.constraints[0].target = c2


def build_all():
    clear_scene()
    build_world()
    build_room()
    sacks = build_racks()
    build_dock()
    drone = build_drone()
    build_callouts(drone)
    special = build_scan_fx(drone, sacks)
    build_panel()
    animate_drone(drone)
    build_cameras(drone)
    bpy.context.scene.frame_set(1)
    print(f"built scene: {len(sacks)} sacks, statuses {special}")


if __name__ == "__main__":
    build_all()
