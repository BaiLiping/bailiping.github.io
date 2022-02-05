---
layout: page
title: Beihang HPC
---

----
- TOC
{:toc}
----


# Beihang HPC
<embed src="/assets/docs/Compute/HPC.pdf" type="application/pdf" width="100%" height=1000>
<embed src="/assets/docs/Compute/HPC1.pdf" type="application/pdf" width="100%" height=1000>
<embed src="/assets/docs/Compute/HPC2.pdf" type="application/pdf" width="100%" height=480>
<embed src="/assets/docs/Compute/HPC3.pdf" type="application/pdf" width="100%" height=480>
<embed src="/assets/docs/Compute/HPC4.pdf" type="application/pdf" width="100%" height=480>
<embed src="/assets/docs/Compute/HPC5.pdf" type="application/pdf" width="100%" height=480>
<embed src="/assets/docs/Compute/HPC6.pdf" type="application/pdf" width="100%" height=480>
<embed src="/assets/docs/Compute/HPC7.pdf" type="application/pdf" width="100%" height=480>
<embed src="/assets/docs/Compute/HPC8.pdf" type="application/pdf" width="100%" height=480>
<embed src="/assets/docs/Compute/HPC9.pdf" type="application/pdf" width="100%" height=480>




# Jupyter Interface
reference: 
- [Yale HPC](https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/)

what is great about this method is that you can ssh to the compute node directly, instead of staying in the log in node.

```
#!/bin/bash 
#SBATCH -J name_of_the_job 
#SBATCH -p cpu-high
#SBATCH -N 1                 # one node
#SBATCH -n 1                 # one cpu or gpu
#SBATCH -t 5:00              # maximum time
#SBATCH -o job.out           # will print on the terminal if omitted
#SBATCH -e job.err          

XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')
clusterurl="10.212.70.128"

export PATH=$PATH:~/.local/bin

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${clusterurl}
 
 Here is the MobaXterm info:

 Forwarded port:same as remote port
 Remote server: ${node}
 Remote port: ${port}
 SSH server: ${cluster}.${clusterurl}
 SSH login: $user
 SSH port: 22

 Use a Browser on your local machine to go to:
 localhost:${port} (prefix w/ https:// if using password)

 or copy the URL from below and put there localhost after http:// so it would be something like:
 http://localhost:9499/?token=86c93ba16aaead7529a5da0e5e5a46be7ad8cfea35b2d49f
 "

 # load modules or conda environments here
 # e.g. farnam:
 # module load Python/2.7.13-foss-2016b 
 # conda env activate mx
 # DON'T USE ADDRESS BELOW. 
 # DO USE TOKEN BELOWa
 # module load anaconda3
 # source activate blpenv

#srun -p gpu-high --gres=gpu:2 jupyter lab --no-browser --port=${port} --ip=${node}
jupyter lab --no-browser --port=${port} --ip=${node}
```



# DATASETS

1. CARRADA Dataset:
- [paper](https://arxiv.org/abs/2005.01456) 
- [repository](https://github.com/valeoai/carrada_dataset) 
- [data](https://arthurouaknine.github.io/codeanddata/carrada)
2. RADIATE DATASET:
- [paper](https://arxiv.org/abs/2010.09076)
- [repository](https://github.com/marcelsheeny/radiate_sdk)
- [website](http://pro.hw.ac.uk/radiate/)
- [data](http://pro.hw.ac.uk/radiate/downloads/)
3. CRUW Dataset:
- [paper](https://openaccess.thecvf.com/content/WACV2021/html/Wang_RODNet_Radar_Object_Detection_Using_Cross-Modal_Supervision_WACV_2021_paper.html)
- [repository1](https://github.com/yizhou-wang/cruw-devkit)
- [repository2](https://github.com/yizhou-wang/RODNet)
- [website](https://www.cruwdataset.org/introduction)
4. NuScenes Dataset:
- [paper](https://arxiv.org/abs/1903.11027)
- [repository](https://github.com/nutonomy/nuscenes-devkit)
- [website](https://www.nuscenes.org/)
5. RadarScenes Dataset:
- [paper](https://arxiv.org/abs/2104.02493)
- [reporitory](https://github.com/oleschum/radar_scenes)
- [website](https://radar-scenes.com/)
6. RADDet Dataset:
- [paper](https://arxiv.org/abs/2105.00363)
- [repository](https://github.com/ZhangAoCanada/RADDet)
- [googledrive](https://drive.google.com/drive/folders/1v-AF873jP8p6waChF3pSSqz6HXOOZgkC)
7. Real-World Marine Radar Datasets for Target Tracking： (export controlled)
- [website](https://www.dlr.de/kn/en/desktopdefault.aspx/tabid-15772/25571_read-64760/)
8. Waymo Open Dataset:
- [website](https://waymo.com/open/)
9. Coloradar:
- [website](https://arpg.github.io/coloradar//)
- [paper](https://arxiv.org/abs/2103.04510)

## SERVER
- [featurize](https://featurize.cn)
![price](/assets/img/radarproject/featurizeprice.png)
- [matpool](https://www.matpool.com/)
![price](/assets/img/radarproject/matpoolprice.jpg)
- [gpushare](https://www.gpushare.com/)

## WORKSTATION
- Lenovo thinkstation p520
- CPU Xeon w2125 4ghz
- GPU rtx2080 8GMEM
- MEM 32G，can be extended to 128G



### SLURM
<iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries?list=PLyqL4-20ZuTR6k-hwR0hzGp4wvWI377Wy" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>