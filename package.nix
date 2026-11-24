{
  self,
  lib,
  pyproject,
  buildPythonPackage,
  setuptools,
  pytestCheckHook,
  scipy,
  numpy,
  numba,
  scikit-learn,
}:
buildPythonPackage {
  pname = pyproject.project.name;
  version = pyproject.project.version;
  pyproject = true;

  src = self;

  build-system = [ setuptools ];

  dependencies = [
    scipy
    numpy
    numba
    scikit-learn
  ];

  # Deactivate tests as we don't have rpy2 available
  # nativeCheckInputs = [ pytestCheckHook ];

  pythonImportsCheck = [
    "ondil"
  ];

  meta = {
    description = "A package for online distributional learning.";
    homepage = "https://github.com/simon-hirsch/ondil";
    license = lib.licenses.gpl3Only;
    maintainers = with lib.maintainers; [ berrij ];
    platforms = lib.platforms.all;
  };
}
