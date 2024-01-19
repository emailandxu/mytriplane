echo $1
blender-3.2.2-linux-x64/blender \
--background \
--python \
blender_script.py \
-- \
--object_path \
"../data/3d/$1.glb" \
--num_renders \
36 \
--output_dir \
"../data/2d/$1" \
--engine \
BLENDER_EEVEE
