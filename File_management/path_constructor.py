import os

def construct_data_output_path(options_json):
    """
    Constructs a file path based on the structure of options_json.

    Parameters
    ----------
    options_json : dict
        Dictionary containing the metadata and parameters for constructing the file path.

    Returns
    -------
    str
        The constructed file path.
    """
    # Extract information from the "description" section
    description = options_json.get("description", {})
    model = description.get("model", "")
    project = description.get("project", "")
    simulation_ID = description.get("simulation_ID", "")
    grid = description.get("grid", "")
    grid_time_zoom = f"g{grid}_t{description.get('time', '')}_z{description.get('zoom', '')}"
    region = description.get("region", "")

    # Extract information from the "parameters" section
    parameters = options_json.get("parameters", {})
    date_fr = parameters.get("date_fr", "")
    date_to = parameters.get("date_to", "")
    axis1_name = parameters.get("Axis1", {}).get("name", "")
    try:
        axis2_name = parameters.get("Axis2", {}).get("name", "")
    except Exception as details:
        axis2_name = ""
    try:
        axis3_name = parameters.get("Axis3", {}).get("name", "")
    except Exception as details:
        axis3_name = ""

    # Construct the file path
    if axis3_name == '' and axis2_name == '':
        output_file_path = (
            f"/work/mh0066/m301130/CompiledData/{model}_{project}/{simulation_ID}/{grid_time_zoom}/"
            f"{region}/{axis1_name}/{date_fr}_{date_to}"
        )
    elif axis3_name == '':
        output_file_path = (
            f"/work/mh0066/m301130/CompiledData/{model}_{project}/{simulation_ID}/{grid_time_zoom}/"
            f"{region}/{axis1_name}_vs_{axis2_name}/{date_fr}_{date_to}"
        )
    else:
        output_file_path = (
            f"/work/mh0066/m301130/CompiledData/{model}_{project}/{simulation_ID}/{grid_time_zoom}/"
            f"{region}/{axis1_name}_vs_{axis2_name}_vs_{axis3_name}/{date_fr}_{date_to}"
        )
    # Create directory if it doesn't exist
    os.makedirs(output_file_path, exist_ok=True)
    return output_file_path
