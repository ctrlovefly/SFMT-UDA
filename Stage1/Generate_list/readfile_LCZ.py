# 生成数据集对应的txt包含 (数据路径, 标签)
# generate .txt from the LCZ dataset. A line is (image path, true label)
import os
import pdb
import random
folder ='../../../../dataset/city_wise_png_jilin/'# change the path of your dataset

domains = os.listdir(folder)
domains.sort()
print(f'domains:{domains}')
class_common=['1', '2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '17']
print(f'common classes:{class_common}')

for d in range(len(domains)):
	dom = domains[d]
	if os.path.isdir(os.path.join(folder, dom)):
		dom_new = dom.replace(" ","_")
		print(dom, dom_new)
		os.rename(os.path.join(folder, dom), os.path.join(folder, dom_new))

		classes = os.listdir(os.path.join(folder, dom_new))
		classes.sort()
		classes = list(set(classes).intersection(class_common))
		classes = sorted(classes, key=int)

		f = open(dom_new + "_list.txt", "w")
		for c in range(len(classes)):
			cla = classes[c]
			cla_new = cla.replace(" ","_")
			print(cla, cla_new)
			os.rename(os.path.join(folder, dom_new, cla), os.path.join(folder, dom_new, cla_new))
			files = os.listdir(os.path.join(folder, dom_new, cla_new))
			files.sort()

			for file in files:
				file_new = file.replace(" ","_")
				os.rename(os.path.join(folder, dom_new, cla_new, file), os.path.join(folder, dom_new, cla_new, file_new))
				print(file, file_new)
				print('{:} {:}'.format(os.path.join(folder, dom_new, cla_new, file_new), c))

				f.write('{:} {:}\n'.format(os.path.join(dom_new, cla_new, file_new), c))

		f.close()