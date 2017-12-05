import csv

from vtk import *
file = open("iris.csv", "rb")
obj = csv.reader(file.read().decode('utf-8').splitlines())

title = False

data = {}
types = {}

for v in obj:
	if title == False:
		title = True
		continue
	if len(v) != 0:
		if v[3] not in data:
			data[v[3]] = {}
		communities = data[v[3]]
		if v[1] not in communities:
			communities[v[1]] = {}
		zipcodes = communities[v[1]]
		if v[0] not in zipcodes:
			zipcodes[v[0]] = {}
		species = zipcodes[v[0]]
		if v[2] not in species:
			species[v[2]] = [v[4]]
		else:
			species[v[2]].append(v[4])
		if v[2] not in types:
			types[v[2]] = len(types)

file = open('./iris.xml', 'w')

file.write('<vertex name="Borough">\n')

for borough, communities in data.items():
	file.write('   <vertex size="' + str(len(communities)) + '" name="' + borough + '">\n')
	for community, zipcodes in communities.items():
		file.write('      <vertex size="' + str(len(zipcodes)) + '" name="' + community + '">\n')
		for zipcode, species in zipcodes.items():
			file.write('         <vertex size="' + str(len(species)) + '" name="' + zipcode + '">\n')
			for tree, diameters in species.items():
				for diameter in diameters:
				   diameters.sort()
				   file.write('            <vertex size="' + diameter + '" type="'+ str(types[tree]) + '" name="' + tree + '" />\n')
			file.write('         </vertex>\n')
		file.write('      </vertex>\n')
	file.write('   </vertex>\n')

file.write('</vertex>')

file.close()



reader1 = vtkXMLTreeReader()
reader1.SetFileName("iris.xml")
reader1.Update()

numeric = vtkStringToNumeric()
numeric.SetInputConnection(reader1.GetOutputPort())

view = vtkTreeMapView()
view.SetAreaSizeArrayName("size");
view.SetAreaColorArrayName("type");
view.SetAreaLabelArrayName("name");
view.SetAreaLabelVisibility(True);
view.SetAreaHoverArrayName("name");
view.SetLayoutStrategyToSquarify();
view.SetRepresentationFromInputConnection(numeric.GetOutputPort());

# Apply a theme to the views
theme = vtkViewTheme.CreateNeonTheme()
view.ApplyViewTheme(theme)
theme.FastDelete()

view.ResetCamera()
view.Render()

view.GetInteractor().Start()
