var documenterSearchIndex = {"docs":
[{"location":"tutorials/using_mlj/#Working-with-MLJ","page":"Example 2: Working with MLJ","title":"Working with MLJ","text":"","category":"section"},{"location":"api/#Function-Documentation","page":"Function Docs","title":"Function Documentation","text":"","category":"section"},{"location":"api/","page":"Function Docs","title":"Function Docs","text":"Documentation for SelfOrganizingMaps.","category":"page"},{"location":"api/","page":"Function Docs","title":"Function Docs","text":"","category":"page"},{"location":"api/","page":"Function Docs","title":"Function Docs","text":"Modules = [SelfOrganizingMaps]","category":"page"},{"location":"api/#SelfOrganizingMaps.SelfOrganizingMap","page":"Function Docs","title":"SelfOrganizingMaps.SelfOrganizingMap","text":"SelfOrganizingMap\n\nA model type for constructing a self organizing map, based on SelfOrganizingMaps.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nSelfOrganizingMap = @load SelfOrganizingMap pkg=SelfOrganizingMaps\n\nDo model = SelfOrganizingMap() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in SelfOrganizingMap(k=...).\n\nSelfOrganizingMaps implements Kohonen's Self Organizing Map, Proceedings of the IEEE; Kohonen, T.; (1990):\"The self-organizing map\"\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X) where\n\nX: an AbstractMatrix or Table of input features whose columns are of scitype Continuous.\n\nTrain the machine with fit!(mach, rows=...).\n\nHyper-parameters\n\nk=10: Number of nodes along once side of SOM grid. There are k² total nodes.\nη=0.5: Learning rate. Scales adjust made to wining node and its neighbors during each round of training.\nσ²=0.05: The (squared) neighbor radius. Used to determine scale for neighbor node adjustments.\ngrid_type=:rectangular  Node grid geometry. One of (:rectangular, :hexagonal, :spherical).\nη_decay=:exponential Learning rate schedule function. One of (:exponential, :asymptotic)\nσ_decay=:exponential Neighbor radius schedule function. One of (:exponential, :asymptotic, :none)\nneighbor_function=:gaussian Kernel function used to make adjustment to neighbor weights. Scale is set by σ². One of (:gaussian, :mexican_hat).\nmatching_distance=euclidean Distance function from Distances.jl used to determine winning node.\nNepochs=1 Number of times to repeat training on the shuffled dataset.\n\nOperations\n\ntransform(mach, X): returns the coordinates of the winning SOM node for each instance of X\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nweights: Array of weight vectors for the SOM nodes.\ncoords: The coordinates of each of the SOM nodes.\n\nReport\n\nThe fields of report(mach) are:\n\nclasses: the index of the winning node for each instance in X interpreted as a class label\n\nExamples\n\nusing MLJ\nsom = @load SelfOrganizingMap pkg=SelfOrganizingMaps\nmodel = som()\nX, y = make_regression(50, 3) # synthetic data\nmach = machine(model, X) |> fit!\nX̃ = transform(mach, X)\n\nrpt = report(mach)\nclasses = rpt.classes\n\n\n\n\n\n","category":"type"},{"location":"tutorials/colors/#Mapping-Color-to-the-2-D-Plane","page":"Example 1: Colors","title":"Mapping Color to the 2-D Plane","text":"","category":"section"},{"location":"tutorials/colors/","page":"Example 1: Colors","title":"Example 1: Colors","text":"We'll put the example currently in the tests here...","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SelfOrganizingMaps","category":"page"},{"location":"#SelfOrganizingMaps","page":"Home","title":"SelfOrganizingMaps","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SelfOrganizingMaps is a package for creating self organizing maps in julia designed for compatibility with MLJ. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"Documentation for SelfOrganizingMaps.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SelfOrganizingMaps]","category":"page"}]
}
