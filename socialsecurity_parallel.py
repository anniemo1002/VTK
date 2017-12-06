import vtk

inputFilename = "socialsecurity.csv"
title = "socialsecurity"

stateCode = {"AL": 0, "AK": 1, "AZ": 2, "AR": 3, "CA": 4, 
             "CO": 5, "CT": 6, "DE": 7, "FL": 8, "GA": 9, 
             "HI": 10, "ID": 11, "IL": 12, "IN": 13, "IA": 14, 
             "KS": 15, "KY": 16, "LA": 17, "ME": 18, "MD": 19, 
             "MA": 20, "MI": 21, "MN": 22, "MS": 23, "MO": 24, 
             "MT": 25, "NE": 26, "NV": 27, "NH": 28, "NJ": 29, 
             "NM": 30, "NY": 31, "NC": 32, "ND": 33, "OH": 34, 
             "OK": 35, "OR": 36, "PA": 37, "RI": 38, "SC": 39, 
             "SD": 40, "TN": 41, "TX": 42, "UT": 43, "VT": 44, 
             "VA": 45, "WA": 46, "WV": 47, "WI": 48, "WY": 49, 
             "DC": 50, "FM": 51, "GU": 52, "MH": 53, "MP": 54, 
             "PW": 55, "PR": 56, "VI": 57, "AE": 58, "AA": 59, 
             "AP": 60 }

reader = vtk.vtkDelimitedTextReader()
reader.SetFileName(inputFilename)
reader.SetHaveHeaders(1)
reader.DetectNumericColumnsOn()
reader.SetFieldDelimiterCharacters(",")
reader.Update()
table = reader.GetOutput()

states = table.GetColumn(4)
statesNo = vtk.vtkIntArray()
statesNo.SetName("State No")
statesNo.SetNumberOfComponents(1)
statesNo.SetNumberOfTuples(states.GetNumberOfValues())

for i in range(0, states.GetNumberOfValues()):
    statesNo.SetTuple1(i, stateCode[states.GetValue(i).strip()])

table.AddColumn(statesNo)

lookup = vtk.vtkLookupTable()
lookup.SetNumberOfTableValues(61)
lookup.Build()
for i in range(0, 61):
    lookup.SetTableValue(i, (i%5)/5.00, (i%55)/55.00, (i%3)/3.00)
lookup.SetRange(0, 61)
lookup.SetAlpha(1.0)

chart = vtk.vtkChartParallelCoordinates()
chart.GetPlot(0).SetInputData(table)
chart.SetColumnVisibilityAll(True)
chart.SetColumnVisibility("File Name", False)
chart.SetColumnVisibility("File Version", False)
chart.SetColumnVisibility("Update Date", False)
chart.SetColumnVisibility("Region Code", False)
chart.SetColumnVisibility("Date Type", False)
chart.SetColumnVisibility("Percent of Adult Population Receiving SSA Adult Disability Benefits", False)
chart.SetColumnVisibility("Eligible Adult Population Filing Rate", False)
chart.SetColumnVisibility("Eligible Adult Population Allowance Rate", False)
chart.SetColumnVisibility("Adult Favorable  Determination Rate", False)
chart.SetColumnVisibility("Percent of Population under age 18 Receiving SSI DC Benefits", False)
chart.SetColumnVisibility("Eligible Child Population Filing Rate", False)
chart.SetColumnVisibility("Eligible Child Population Allowance Rate", False)
chart.SetColumnVisibility("SSI Disabled Child Allowance Rate", False)
chart.SetColumnVisibility("Favorable Determination Rate", False)
chart.SetColumnVisibility("Date", False)
chart.SetColumnVisibility("State No", False)
chart.SetColumnVisibility("State Code", False)
chart.SetColumnVisibility("Favorable Adult Determinations", False)
chart.SetColumnVisibility("SSI Disabled Child (DC) Beneficiaries", False)
chart.SetColumnVisibility("Favorable Determination Rate", False)
chart.SetColumnVisibility("All Favorable Determinations", False)
chart.SetColumnVisibility("All Determinations", False)

chart.GetPlot(0).SetScalarVisibility(1)
chart.GetPlot(0).SetLookupTable(lookup)
chart.GetPlot(0).SelectColorArray("State No")
chart.GetPlot(0).GetPen().SetOpacityF(0.8)

view = vtk.vtkContextView()
view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
view.GetRenderWindow().SetSize(1800, 900)
view.GetScene().AddItem(chart)
view.GetRenderWindow().SetMultiSamples(0)
view.GetRenderWindow().Render()
view.GetInteractor().Start()
