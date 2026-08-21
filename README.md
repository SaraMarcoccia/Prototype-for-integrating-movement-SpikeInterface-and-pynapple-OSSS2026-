From osss-collaboration-days (Niko)

Idea description
This is an idea we've discussed in passing with @chrishalcrow and @wulfdewolf.

SpikeInterface is a great open-source tool for processing extracellular electrophysiology data, and is taught as part of that track on week 2.
pynapple is a great neural data analysis package, also taught as part of the extracellular electrophysiology track on week 2.
movement is our Python package for processing motion tracking data, and is taught as part of the "Animals in Motion" track on week 1.
I think there is an opportunity here to draft an example workflow/notebook that integrates all three.


In broad strokes:

Load processed ephys data (from a SpikeInterface analysis object?)
Load motion timeseries (from a movement netCDF file)?
Align the two in time
Use pynapple to compute something that meaningfully needs both sources of information (e.g. place fields or head direction tuning).
The end-product could be just a Jupyter notebook, or a proper example to be published on the website(s) of one of these project.
