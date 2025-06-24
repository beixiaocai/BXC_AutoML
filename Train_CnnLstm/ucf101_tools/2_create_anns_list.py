import os 
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

	print("开始执行")

	ucf_dir='E:\\ai\\datasets\\UCF-101'
	abs_path = "E:\\ai\\datasets\\UCF-101_cnn_lstm"
	annotation_folder='E:\\ai\\datasets\\UCF-101_cnn_lstm\\annotation'

	if not os.path.exists(annotation_folder):
		os.makedirs(annotation_folder)


	# cleans .txt files
	open(os.path.join(abs_path,'classInd.txt'),'w').write("")
	open(os.path.join(abs_path,'trainlist01.txt'),'w').write("")
	open(os.path.join(abs_path,'testlist01.txt'),'w').write("")
	open(os.path.join(abs_path,'trainval.txt'),'w').write("")

	# updates labels.txt
	# fight 0
	# noFight 1
	labels=os.listdir(ucf_dir)
	for i,label in enumerate(labels):
		with open(os.path.join(abs_path,'classInd.txt'),'a') as f:
			f.write(str(i)+" "+label)
			f.write('\n')

	# loading mapping...
	dict_labels={}
	a=open(os.path.join(abs_path,'classInd.txt'),'r').read()
	c=a.split('\n')
	for i in c[:len(c)-1]:
		dict_labels.update({i.split(' ')[1]:i.split(' ')[0]})

	# generating trainval.txt
	for i,label in enumerate(labels):
		vid_names=os.listdir(os.path.join(ucf_dir,label))
		for video_name in vid_names:
			with open(os.path.join(abs_path,'trainval.txt'),'a') as f:
				f.write(os.path.join(label,video_name)+ " " + dict_labels[label])
				f.write('\n')

	X = []
	y=[]
	for data in open(os.path.join(abs_path,'trainval.txt'),'r').read().split("\n"):
		data_path = data.split()
		if data_path:
			img_path = data_path[0]
			label = data_path[1]
			X.append(img_path)
			y.append(label)

	# split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# generating train.txt
	for i,j in zip(X_train,y_train):
		with open(os.path.join(abs_path,'trainlist01.txt'),'a') as f:
			f.write(i+" "+j)
			f.write('\n')

	# generating test.txt
	for i,j in zip(X_test,y_test):
		with open(os.path.join(abs_path,'testlist01.txt'),'a') as f:
			f.write(i)
			f.write('\n')


	def copy_file(src_filepath,dst_filepath):
		rf = open(src_filepath, "rb")
		content = rf.read()
		rf.close()

		wf = open(dst_filepath, "wb")
		wf.write(content)
		wf.close()

		os.remove(src_filepath)

	copy_file(src_filepath=os.path.join(abs_path,"classInd.txt"),
			  dst_filepath=os.path.join(annotation_folder,"classInd.txt"))

	copy_file(src_filepath=os.path.join(abs_path,"testlist01.txt"),
			  dst_filepath=os.path.join(annotation_folder,"testlist01.txt"))

	copy_file(src_filepath=os.path.join(abs_path,"trainlist01.txt"),
			  dst_filepath=os.path.join(annotation_folder,"trainlist01.txt"))

	os.remove(os.path.join(abs_path,"trainval.txt"))

	print("正常结束")

