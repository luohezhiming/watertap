How to use a property model
------------------------------------------------

The example below shows how to use a property model and display outputs for a state block. Property models allow
users to model the chemical and physical properties of simple systems without the use of unit models.

.. testsetup::

   # quiet idaes logs
   import idaes.logger as idaeslogger
   idaeslogger.getLogger('ideas.core').setLevel('CRITICAL')
   idaeslogger.getLogger('idaes.init').setLevel('CRITICAL')

.. doctest::

    # Import concrete model from Pyomo
    from pyomo.environ import ConcreteModel
    # Import flowsheet block from IDAES core
    from idaes.core import FlowsheetBlock
    # Import solver from IDAES core
    from idaes.core.solvers import get_solver
    # Import NaCl property model
    import watertap.property_models.NaCl_prop_pack as props
    # Import utility tool for calculating scaling factors
    import idaes.core.util.scaling as iscale


    # Create a concrete model, flowsheet, and NaCl property parameter block.
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.properties = props.NaClParameterBlock()


    # Build the state block and specify a time (0 = steady state).
    m.fs.state_block = m.fs.properties.build_state_block([0], default={})

    # Fully specify the system.
    feed_flow_mass = 1
    feed_mass_frac_NaCl = 0.035
    feed_mass_frac_H2O = 1 - feed_mass_frac_NaCl
    feed_pressure = 50e5
    feed_temperature = 298.15

    m.fs.state_block[0].flow_mass_phase_comp['Liq', 'NaCl'].fix(feed_flow_mass * feed_mass_frac_NaCl)
    m.fs.state_block[0].flow_mass_phase_comp['Liq', 'H2O'].fix(feed_flow_mass * feed_mass_frac_H2O)
    m.fs.state_block[0].pressure.fix(feed_pressure)
    m.fs.state_block[0].temperature.fix(feed_temperature)

    # Set scaling factors for component mass flowrates (variable * scaling factor should be between 0.01 and 100).
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'NaCl'))
    iscale.calculate_scaling_factors(m.fs)

    # "Touch" build-on-demand variables so that they are created. If these are not touched before running the solver, the output would only display their initial values, not their actual values.
    m.fs.state_block[0].dens_mass_phase['Liq']
    m.fs.state_block[0].conc_mass_phase_comp['Liq', 'NaCl']
    m.fs.state_block[0].flow_vol_phase['Liq']
    m.fs.state_block[0].molality_comp['NaCl']
    m.fs.state_block[0].visc_d_phase['Liq']
    m.fs.state_block[0].diffus_phase['Liq']
    m.fs.state_block[0].enth_mass_phase['Liq']
    m.fs.state_block[0].pressure_osm

    # Create the solver object.
    solver = get_solver()

    # Solve the model and display the output.
    results = solver.solve(m, tee=False)
    #m.fs.state_block[0].display()

