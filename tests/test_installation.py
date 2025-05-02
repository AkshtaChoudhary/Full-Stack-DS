def test_package_import():
    try:
        import housing
    except ImportError:
        assert False, "housing package not installed properly"
