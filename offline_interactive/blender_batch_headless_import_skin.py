import argparse
import sys
from pathlib import Path
import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    ap = argparse.ArgumentParser(description="Headless Blender mesh import + optional Skin modifier")
    ap.add_argument("--mesh", required=True, help="Input mesh path (.obj/.ply/.stl/.glb/.gltf)")
    ap.add_argument("--save-blend", default=None, help="Optional .blend output path")
    ap.add_argument("--export-obj", default=None, help="Optional exported OBJ path")
    ap.add_argument("--skin", type=int, default=1, help="1 to add Blender Skin modifier, 0 to skip")
    ap.add_argument("--subsurf", type=int, default=1, help="1 to add Subsurf after Skin, 0 to skip")
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_mesh(filepath: str):
    ext = Path(filepath).suffix.lower()
    before = set(bpy.data.objects)

    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    elif ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    after = set(bpy.data.objects)
    imported = list(after - before)
    meshes = [obj for obj in imported if obj.type == 'MESH']
    if not meshes:
        raise RuntimeError("No mesh objects were imported")
    return meshes


def join_meshes(meshes):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def prep_object(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.shade_smooth()


def add_skin_stack(obj, use_skin: bool, use_subsurf: bool):
    bpy.context.view_layer.objects.active = obj
    if use_skin:
        obj.modifiers.new(name="Skin", type='SKIN')
    if use_skin and use_subsurf:
        subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
        subsurf.levels = 2
        subsurf.render_levels = 2


def export_obj(filepath: str, obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True)


def main():
    args = parse_args()
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    clear_scene()
    meshes = import_mesh(str(mesh_path))
    obj = join_meshes(meshes)
    obj.name = "ImportedMesh"
    prep_object(obj)
    add_skin_stack(obj, bool(args.skin), bool(args.subsurf))

    print(f"Imported object: {obj.name}")
    print(f"Vertex count: {len(obj.data.vertices)}")
    print(f"Face count: {len(obj.data.polygons)}")
    print(f"Skin modifier: {bool(args.skin)}")
    print(f"Subsurf modifier: {bool(args.subsurf)}")

    if args.export_obj:
        export_obj(args.export_obj, obj)
        print(f"Exported OBJ to: {args.export_obj}")

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
        print(f"Saved blend file to: {args.save_blend}")


if __name__ == "__main__":
    main()
