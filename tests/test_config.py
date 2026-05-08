import os

import neat


def test_nonexistent_config():
    """Check that attempting to open a non-existent config file raises
    an Exception with appropriate message."""
    passed = False
    try:
        c = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'wubba-lubba-dub-dub')
    except Exception as e:
        passed = 'No such config file' in str(e)
    assert passed

def test_bad_config_default_activation():
    """Check that an activation function default not in the list of options
    raises an Exception."""
    test_bad_config_RuntimeError(config_file='bad_configuration0')

def test_bad_config_unknown_option():
    """Check that an unknown option (at least in some sections) raises an exception."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'bad_configuration2')
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except NameError:
        pass
    else:
        raise Exception("Did not get a NameError from an unknown configuration file option (in the 'DefaultSpeciesSet' section)")
    config3_path = os.path.join(local_dir, 'bad_configuration3')
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config3_path)
    except NameError:
        pass
    else:
        raise Exception("Did not get a NameError from an unknown configuration file option (in the 'NEAT' section)")


def test_bad_config_RuntimeError(config_file='bad_configuration4'):
    """Test for RuntimeError with a bad configuration file."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except RuntimeError:
        pass
    else:
        raise Exception(
            "Should have had a RuntimeError with {!s}".format(config_file))


def test_bad_config5():
    """Test using bad_configuration5 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration5')


def test_bad_config6():
    """Test using bad_configuration6 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration6')


def test_bad_config7():
    """Test using bad_configuration7 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration7')


def test_bad_config8():
    """Test using bad_configuration8 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration8')


def test_bad_config9():
    """Test using bad_configuration9 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration9')


def test_nonpositive_time_constant_min_value_raises(tmp_path):
    """``time_constant_min_value <= 0`` causes ``exp(-dt/tau)`` to blow up
    in CTRNN.advance; reject it at config-load time."""
    local_dir = os.path.dirname(__file__)
    src = os.path.join(local_dir, 'test_configuration_gpu_ctrnn')
    with open(src) as f:
        content = f.read()
    for bad in ('0.0', '-0.1'):
        bad_content = content.replace(
            'time_constant_min_value    = 0.01',
            f'time_constant_min_value    = {bad}')
        bad_path = tmp_path / f'tc_min_{bad}'
        bad_path.write_text(bad_content)
        try:
            neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        str(bad_path))
        except RuntimeError as e:
            assert 'time_constant_min_value' in str(e)
        else:
            raise AssertionError(
                f"Expected RuntimeError for time_constant_min_value = {bad}")


def test_unknown_genome_attribute_key_raises(tmp_path):
    """A typo in a [DefaultGenome] gene-attribute key (e.g.
    ``bias_init_typ`` instead of ``bias_init_type``) must raise
    UnknownConfigItemError instead of being silently dropped and
    backfilled with defaults.
    """
    local_dir = os.path.dirname(__file__)
    src = os.path.join(local_dir, 'test_configuration')
    with open(src) as f:
        content = f.read().replace('bias_init_type', 'bias_init_typ')
    typo_path = tmp_path / 'typo_config'
    typo_path.write_text(content)
    try:
        neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    str(typo_path))
    except neat.config.UnknownConfigItemError as e:
        assert 'bias_init_typ' in str(e)
    else:
        raise AssertionError(
            "Expected UnknownConfigItemError for typo 'bias_init_typ' in [DefaultGenome] section")


if __name__ == '__main__':
    test_nonexistent_config()
    # test_bad_config_activation()
    test_bad_config_unknown_option()
    test_bad_config_RuntimeError()
    test_bad_config5()
    test_bad_config6()
    test_bad_config7()
    test_bad_config8()
    test_bad_config9()
