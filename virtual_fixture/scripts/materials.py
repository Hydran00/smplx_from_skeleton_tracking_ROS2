import open3d as o3d
mat_sphere_transparent = o3d.visualization.rendering.MaterialRecord()
mat_sphere_transparent.shader = 'defaultLitTransparency'
mat_sphere_transparent.base_color = [0.467, 0.467, 0.467, 0.6]
mat_sphere_transparent.base_roughness = 0.0
mat_sphere_transparent.base_reflectance = 0.0
mat_sphere_transparent.base_clearcoat = 1.0
mat_sphere_transparent.thickness = 1.0
mat_sphere_transparent.transmission = 1.0
mat_sphere_transparent.absorption_distance = 10
mat_sphere_transparent.absorption_color = [0.5, 0.5, 0.5]

mat_skin = o3d.visualization.rendering.MaterialRecord()
mat_skin.shader = 'defaultLitTransparency'
mat_skin.base_color = [0.467, 0.467, 0.467, 0.6]
mat_skin.base_roughness = 0.0
mat_skin.base_reflectance = 0.0
mat_skin.base_clearcoat = 1.0
mat_skin.thickness = 1.0
mat_skin.transmission = 1.0
mat_skin.absorption_distance = 10
mat_skin.absorption_color = [0.5, 0.5, 0.8]