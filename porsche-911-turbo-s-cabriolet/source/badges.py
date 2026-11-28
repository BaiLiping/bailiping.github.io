"""Raised emblems on the 992.2 Turbo S Cabriolet, laid onto the corresponding part.

The rear PORSCHE capitals and 911 plate were traced from the configured rear frame
(model 992652, Porsche vehicle imagery). The two scripts and the monochrome crest
were sampled from the Porsche Newsroom's Turbo S Cabriolet launch photographs:
https://newsroom.porsche.com/dam/jcr:3935afed-0bb8-4c19-9350-a687ed1886a8/911%20Turbo%20S%20Cabrio%20Provence%2030.jpg
https://newsroom.porsche.com/dam/jcr:6c9879dd-2a39-4292-a9dc-b9d6fae57566/911%20Turbo%20S%20Cabrio%20Provence%2034.jpg
https://newsroom.porsche.com/dam/jcr:39490023-b6d8-4bd6-8cc6-463f74a9f0ba/911%20Turbo%20S%20Cabrio%20Ice%20Grey%20Metallic%2022.jpg
"""
exec(compile((R/'source/badge_profiles.py').read_text(),str(R/'source/badge_profiles.py'),'exec'))

turbonite_lettering=mat('Raised Turbonite lettering',(.31,.296,.262),.73,.28)
crest_enamel=mat('Turbonite shield enamel and stamped relief',(.32,.32,.30),.36,.34)
crest_face=crest_enamel.node_tree.nodes.new('ShaderNodeTexImage')
crest_face.image=bpy.data.images.load(str(R/'source/turbonite-crest.png'),check_existing=True)
crest_face.image.pack()
crest_enamel.node_tree.links.new(crest_face.outputs['Color'],
                                 crest_enamel.node_tree.nodes['Principled BSDF'].inputs['Base Color'])

def badge_relief(name,profile,width,height,project,material,thickness):
    # A single flat curve preserves detached glyphs and their counters when filled.
    curve=bpy.data.curves.new(name,'CURVE')
    curve.dimensions='2D'
    curve.fill_mode='BOTH'
    curve.extrude=thickness*.5
    for ring in BADGE_PROFILES[profile]:
        poly=curve.splines.new('POLY')
        poly.points.add(len(ring)-1)
        for point,(u,v) in zip(poly.points,ring):
            point.co=(u*width,v*height,0,1)
        poly.use_cyclic_u=True
    temporary=bpy.data.objects.new(name,curve)
    bpy.context.scene.collection.objects.link(temporary)
    evaluated=temporary.evaluated_get(bpy.context.evaluated_depsgraph_get())
    geometry=bpy.data.meshes.new_from_object(evaluated)
    bpy.data.objects.remove(temporary,do_unlink=True)
    bpy.data.curves.remove(curve)
    for vertex in geometry.vertices:
        u,v,d=vertex.co
        vertex.co=cv(project(u,v,d+thickness*.5))
    geometry.materials.append(material)
    geometry.update()
    obj=bpy.data.objects.new(name,geometry)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent=group(parent)
    return obj

def badge_crest(name,width,height,project,owner):
    # The perimeter is the actual shield outline. Its stamped rim and crowned face
    # remain solid when viewed edge-on. UVs follow the rectified enamel photograph.
    rim=BADGE_PROFILES['turbonite-crest'][0]
    n=len(rim)
    depth=width*.031
    points=[project(0,0,depth*1.30)]
    texcoords=[(.5,.5)]
    for scale,lift in [(.55,1.2),(.94,1.07),(1,.61),(.99,0)]:
        for u,v in rim:
            points.append(project(u*width*scale,v*height*scale,depth*lift))
            texcoords.append(((u*scale+.5)*509/512,1-(.5-v*scale)*661/672))
    faces=[(0,1+i,1+(i+1)%n) for i in range(n)]
    for j in range(3):
        for i in range(n):
            a=1+j*n+i;b=1+j*n+(i+1)%n
            faces.append((a,a+n,b+n,b))
    obj=mesh(name,points,faces,crest_enamel)
    obj.parent=group(owner)
    obj.data.materials.append(turbonite_lettering)
    layer=obj.data.uv_layers.new(name='Enamel photograph')
    for poly in obj.data.polygons:
        if poly.index>=3*n:poly.material_index=1
        for li in poly.loop_indices:layer.data[li].uv=texcoords[obj.data.loops[li].vertex_index]
    return obj
