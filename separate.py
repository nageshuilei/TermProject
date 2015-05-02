import csv
a=None



def linetorow(line):
	line=line.replace(",","")
	line=line.replace('\n','') 
	#print line
	row=[]
	for s in line:
		row.append(s)
	return row

content=[]

with open("testdata") as f:
	content=f.readlines()

# content=content[1:]

with open("testdatanew.csv",'wb') as csvfile:
	a=csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

	for line in content:
		row=linetorow(line)
		#print row
		a.writerow(row)





