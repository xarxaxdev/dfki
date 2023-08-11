Took me a while to find the specific options for this to work, since I had to hop between nvidia/docker/apptainer docs quite a bit.

Anyways, to run with gpu an apptainer container you should 
Build the image
$apptainer build token-classifier-pytorch.sif recipe.def

Run the image with --nv flag
$apptainer run --nv token-classifier-pytorch.sif