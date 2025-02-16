#!/usr/bin/env python3

print("Testing vtk..")
import vtk

sphere_source = vtk.vtkSphereSource()
sphere_source.SetCenter(0.0, 0.0, 0.0)
sphere_source.SetRadius(5.0)
sphere_source.SetThetaResolution(32)
sphere_source.SetPhiResolution(32)
sphere_source.Update()

# Create a mapper that will take the sphere geometry and convert it to graphics primitives
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(sphere_source.GetOutputPort())

# Create an actor that represents the sphere (geometry + properties)
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a renderer and add the actor to it
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.4)  # Set a nice background color

# Create a render window and add the renderer to it
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)
render_window.SetWindowName("Simple VTK Sphere")

# Create an interactor that allows user interaction with the window
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Initialize the interactor and start the rendering loop
# render_window.Render()
# render_window_interactor.Start()


print('vtk OK\n')