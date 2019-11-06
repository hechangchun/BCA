clear
clc 
path = 'F:\BrainAging\SDSU\test\DTIImg';
mskPath = 'F:\BrainAging\SDSU\test\structure_Resliced_SDSU\'
temp = dir(path);
temp = temp(3:end);
for i = 1 : length(temp)
    dpath = [path,'\',temp(i).name];
    copyfile([mskPath,'Resliced_SDSU_level_2514.nii'],dpath);
end