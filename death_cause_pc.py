from vtk import *
import sys
import random
table = vtkTable()
filename = "death_cause.csv"
reader = vtkDelimitedTextReader()
reader.SetFileName(filename)
reader.SetHaveHeaders(1)
reader.DetectNumericColumnsOn()
reader.SetFieldDelimiterCharacters(",")
reader.Update()
table = reader.GetOutput()

iris_type = table.GetColumn(2)
table.GetColumn(0).SetName("year")
table.GetColumn(1).SetName("cause name")
table.GetColumn(2).SetName("cause code")
table.GetColumn(3).SetName("state")
table.GetColumn(4).SetName("deaths")
table.GetColumn(5).SetName("age")

numPts = table.GetNumberOfRows()
arrCat = vtkIntArray()
arrCat.SetName("Category_ids")
arrCat.SetNumberOfComponents(1)
arrCat.SetNumberOfTuples(numPts)
for i in range(numPts):
    if iris_type.GetValue(i) =="Unintentional Injuries":
        arrCat.SetTuple1(i, 0)
    elif iris_type.GetValue(i) =="All Causes":
        arrCat.SetTuple1(i, 1)
    elif iris_type.GetValue(i) =="Alzheimer's disease":
        arrCat.SetTuple1(i, 2)
    elif iris_type.GetValue(i) =="Homicide":
        arrCat.SetTuple1(i, 3)
    elif iris_type.GetValue(i) =="Stroke":
        arrCat.SetTuple1(i, 4)
    elif iris_type.GetValue(i) =="Chronic liver disease and cirrhosis":
        arrCat.SetTuple1(i, 5)
    elif iris_type.GetValue(i) =="CLRD":
        arrCat.SetTuple1(i, 6)
    elif iris_type.GetValue(i) =="Diabetes":
        arrCat.SetTuple1(i, 7)
    elif iris_type.GetValue(i) =="Diseases of Heart":
        arrCat.SetTuple1(i, 8)
    elif iris_type.GetValue(i) =="Essential hypertension and hypertensive renal disease":
        arrCat.SetTuple1(i, 9)
    elif iris_type.GetValue(i) =="Influenza and pneumonia":
        arrCat.SetTuple1(i, 10)
    elif iris_type.GetValue(i) =="Cancer":
        arrCat.SetTuple1(i, 11)
    elif iris_type.GetValue(i) =="Suicide":
        arrCat.SetTuple1(i, 12)
    elif iris_type.GetValue(i) =="Kidney Disease":
        arrCat.SetTuple1(i, 13)
    elif iris_type.GetValue(i) =="Parkinson's disease":
        arrCat.SetTuple1(i, 14)
    elif iris_type.GetValue(i) =="Pneumonitis due to solids and liquids":
        arrCat.SetTuple1(i, 15)
    elif iris_type.GetValue(i) =="Septicemia":
        arrCat.SetTuple1(i, 16)
table.AddColumn(arrCat)
chart = vtkChartParallelCoordinates()
#chart.SetTitle("Parallel coordinate plot, Fisher's Iris data")
view = vtkContextView()
view.GetRenderer().SetBackground(1.0,1.0,1.0)
view.GetRenderWindow().SetSize(600,300)
view.GetScene().AddItem(chart)

cl = []
for i in range(0,16):
    cl.append([i%3/3.0,i%6/6.0,i%7/7.0])
lut = vtkLookupTable()
lutNum = len(cl)
lut.SetNumberOfTableValues(lutNum)
lut.SetRange(0,16);
lut.Build()
for ii,cc in enumerate(cl):
    lut.SetTableValue(ii,cc[0],cc[1],cc[2],1.0)
lut.SetAlpha(1.0)
chart.GetPlot(0).SetInputData(table)
chart.GetPlot(0).SetScalarVisibility(1)
chart.GetPlot(0).SetLookupTable(lut)
chart.GetPlot(0).SelectColorArray("Category_ids")
chart.GetPlot(0).GetPen().SetOpacityF(0.8)
chart.SetColumnVisibility("cause code",False)
chart.SetColumnVisibility("cause name",False)
chart.SetColumnVisibility("state",False)
view.GetRenderWindow().SetMultiSamples(0)
view.GetRenderWindow().Render()
view.GetInteractor().Start()
