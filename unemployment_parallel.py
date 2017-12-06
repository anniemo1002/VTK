import vtk

inputFilename = "unemployment.csv"
title = "unemployment"

reader = vtk.vtkDelimitedTextReader()
reader.SetFileName(inputFilename)
reader.SetHaveHeaders(1)
reader.DetectNumericColumnsOn()
reader.SetFieldDelimiterCharacters(",")
reader.Update()
table = reader.GetOutput()

lookup = vtk.vtkLookupTable()
lookup.SetNumberOfTableValues(78)
lookup.Build()
for i in range(0, 78):
	lookup.SetTableValue(i, (i%7)/7.00, (i%3)/3.00, (i%4)/4.00)
lookup.SetRange(0, 78)
lookup.SetAlpha(1.0)

chart = vtk.vtkChartParallelCoordinates()
chart.GetPlot(0).SetInputData(table)
chart.SetColumnVisibilityAll(True)
chart.SetColumnVisibility("LAUS Area Code ", False)
chart.SetColumnVisibility("State FIPS Code", False)
chart.SetColumnVisibility("County FIPS Code", False)
chart.SetColumnVisibility("Area Title", False)
chart.SetColumnVisibility("State", False)
chart.SetColumnVisibility("Period", False)
chart.SetColumnVisibility("Date", False)

chart.GetPlot(0).SetScalarVisibility(1)
chart.GetPlot(0).SetLookupTable(lookup)
chart.GetPlot(0).SelectColorArray("County FIPS Code")
chart.GetPlot(0).GetPen().SetOpacityF(0.8)

view = vtk.vtkContextView()
view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
view.GetRenderWindow().SetSize(1800, 900)
view.GetScene().AddItem(chart)
view.GetRenderWindow().SetMultiSamples(0)
view.GetRenderWindow().Render()
view.GetInteractor().Start()