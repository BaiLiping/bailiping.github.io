"""Calibrated linear RGB and clear-coat properties for the FK8 export."""

from typing import NamedTuple


class Finish(NamedTuple):
    color: tuple
    roughness: float
    coat: float
    coat_roughness: float


BODY_PAINT = 'Mat_Boost_Blue_B637P'
POCKET_PAINT = 'Mat_Boost_Blue_Pocket'
WHEEL_PAINT = 'Mat_Wheel_Matte_Black'

FINISHES = {
    BODY_PAINT: Finish((0.0, 0.13, 0.49), 0.36, 0.28, 0.21),
    POCKET_PAINT: Finish((0.0, 0.052, 0.196), 0.36, 0.28, 0.21),
    WHEEL_PAINT: Finish((0.018, 0.019, 0.022), 0.32, 0.16, 0.25),
}


def set_blender_finish(material):
    finish = FINISHES[material.name]
    inputs = next(node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED').inputs
    inputs['Base Color'].default_value = (*finish.color, 1)
    inputs['Metallic'].default_value = 0
    inputs['Roughness'].default_value = finish.roughness
    inputs['Coat Weight'].default_value = finish.coat
    inputs['Coat Roughness'].default_value = finish.coat_roughness
    if material.name in (BODY_PAINT, POCKET_PAINT):
        inputs['IOR'].default_value = 1.45
        inputs['Specular IOR Level'].default_value = 0.46


def set_gltf_finishes(document):
    for material in document['materials']:
        finish = FINISHES.get(material['name'])
        if finish is None:
            continue
        material.setdefault('pbrMetallicRoughness', {}).update(
            baseColorFactor=[*finish.color, 1], roughnessFactor=finish.roughness, metallicFactor=0)
        extensions = material.setdefault('extensions', {})
        extensions.setdefault('KHR_materials_clearcoat', {}).update(
            clearcoatFactor=finish.coat, clearcoatRoughnessFactor=finish.coat_roughness)
        if material['name'] in (BODY_PAINT, POCKET_PAINT):
            extensions['KHR_materials_ior'] = {'ior': 1.45}
            extensions['KHR_materials_specular'] = {'specularFactor': 0.92}
        used = document.setdefault('extensionsUsed', [])
        for extension in extensions:
            if extension not in used:
                used.append(extension)
