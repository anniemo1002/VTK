import vtk

inputFilename = "sex_offenders.csv"
title = "sex_offenders"

reader = vtk.vtkDelimitedTextReader()
reader.SetFileName(inputFilename)
reader.SetHaveHeaders(1)
reader.DetectNumericColumnsOn()
reader.SetFieldDelimiterCharacters(",")
reader.Update()
table = reader.GetOutput()

species = table.GetColumn(4)

speciesNo = vtk.vtkIntArray()
speciesNo.SetName("Species_No")
speciesNo.SetNumberOfComponents(1)
speciesNo.SetNumberOfTuples(species.GetNumberOfValues())

for i in range(0, species.GetNumberOfValues()):
	if species.GetValue(i) == "BLACK":
		speciesNo.SetTuple1(i, 0)
	elif species.GetValue(i) == "WHITE":
		speciesNo.SetTuple1(i, 1)
	elif species.GetValue(i) == "WHITE HISPANIC":
		speciesNo.SetTuple1(i, 2)
	elif species.GetValue(i) == "ASIAN/PACIFIC ISLANDER":
		speciesNo.SetTuple1(i, 3)
	elif species.GetValue(i) == "BLACK HISPANIC":
		speciesNo.SetTuple1(i, 4)

table.AddColumn(speciesNo)

lookup = vtk.vtkLookupTable()
lookup.SetNumberOfTableValues(5)
lookup.Build()
lookup.SetTableValue(0, 1, 0, 0)
lookup.SetTableValue(1, 0, 0, 0)
lookup.SetTableValue(0, 0, 1, 0)
lookup.SetTableValue(0, 1, 1, 0)
lookup.SetTableValue(1, 1, 0, 0)
lookup.SetRange(0, 4)
lookup.SetAlpha(1.0)

chart = vtk.vtkChartParallelCoordinates()
chart.GetPlot(0).SetInputData(table)
chart.SetColumnVisibility("Species_No", False)
chart.SetColumnVisibility("LAST", False)
chart.SetColumnVisibility("FIRST", False)
chart.SetColumnVisibility("BLOCK", False)
chart.SetColumnVisibility("BIRTH DATE", False)
chart.SetColumnVisibility("RACE", False)
chart.GetPlot(0).SetScalarVisibility(1)
chart.GetPlot(0).SetLookupTable(lookup)
chart.GetPlot(0).SelectColorArray("Species_No")
chart.GetPlot(0).GetPen().SetOpacityF(0.8)

view = vtk.vtkContextView()
view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
view.GetRenderWindow().SetSize(1800, 900)
view.GetScene().AddItem(chart)
view.GetRenderWindow().SetMultiSamples(0)
view.GetRenderWindow().Render()
view.GetInteractor().Start()
