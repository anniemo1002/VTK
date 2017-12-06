import vtk

inputFilename = "diabetes.csv"
title = "diabetes"

reader = vtk.vtkDelimitedTextReader()
reader.SetFileName(inputFilename)
reader.SetHaveHeaders(1)
reader.DetectNumericColumnsOn()
reader.SetFieldDelimiterCharacters(",")
reader.Update()
table = reader.GetOutput()

#species = table.GetColumn(table.GetNumberOfColumns()-1)

#speciesNo = vtk.vtkIntArray()
#speciesNo.SetName("Species_No")
#speciesNo.SetNumberOfComponents(1)
#speciesNo.SetNumberOfTuples(species.GetNumberOfValues())
'''
for i in range(0, species.GetNumberOfValues()):
	if species.GetValue(i) == "setosa":
		speciesNo.SetTuple1(i, 0)
	elif species.GetValue(i) == "versicolor":
		speciesNo.SetTuple1(i, 1)
	elif species.GetValue(i) == "virginica":
		speciesNo.SetTuple1(i, 2)

table.AddColumn(speciesNo)
'''
lookup = vtk.vtkLookupTable()
lookup.SetNumberOfTableValues(3)
lookup.Build()
lookup.SetTableValue(0, 0, 0, 1)
lookup.SetTableValue(1, 1, 0, 0)
lookup.SetTableValue(2, 0, 1, 0)
lookup.SetRange(0, 564500)
lookup.SetAlpha(1.0)

chart = vtk.vtkChartParallelCoordinates()
chart.GetPlot(0).SetInputData(table)
chart.SetColumnVisibility("Species_No", False)
chart.SetColumnVisibility("Species", False)
chart.GetPlot(0).SetScalarVisibility(1)
chart.GetPlot(0).SetLookupTable(lookup)
chart.GetPlot(0).SelectColorArray("CT")
chart.GetPlot(0).GetPen().SetOpacityF(0.7)

view = vtk.vtkContextView()
view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
view.GetRenderWindow().SetSize(1800, 900)
view.GetScene().AddItem(chart)
view.GetRenderWindow().SetMultiSamples(0)
view.GetRenderWindow().Render()
view.GetInteractor().Start()
