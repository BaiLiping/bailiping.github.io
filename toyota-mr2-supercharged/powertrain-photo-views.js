// Camera poses in the photo’s native frame. Coordinates use the exported glTF axes.
export const powertrainPhotoViews = {
  "4A-GZE Supercharged engine": {
    theta: -3.08923, phi: 0.62207, roll: -0.00333,
    distance: 111.3951, fov: 21.0, center: [-54.7632, 56.1281, 0.1389],
    panels: { engineLid: 0.95433 },
  },
  "Engine bay and raised vented lid": {
    theta: 2.2330314, phi: 1.1584642, roll: -0.1781647,
    distance: 147.715269, fov: 19.224273, center: [-65.179288, 50.888889, 11.52199],
    panels: { engineLid: 1 },
  },
  "Intercooler and clamped charge hoses": {
    theta: -2.35619, phi: 0.80285, roll: -0.18694,
    distance: 51.2013, fov: 25.0, center: [-53.1528, 42.4769, -3.2095],
    panels: { engineLid: 1 },
  },
  "4A-GZE intake plumbing and hoses": {
    theta: -3.07097, phi: 0.59, roll: 0.21636,
    distance: 49.1825, fov: 35.1467, center: [-57.0016, 38.1548, 11.2549],
    panels: { engineLid: 1 },
  },
  "Ignition leads and plug boots": {
    theta: 2.63992, phi: 1.09956, roll: -0.06502,
    distance: 27.1338, fov: 31.0, center: [-56.3851, 41.8525, 11.3121],
    panels: { engineLid: 1 },
  },
  "Rear compartment harness connectors": {
    theta: -3.10669, phi: 0.71558, roll: 0.0976,
    distance: 17.1869, fov: 54.0, center: [-79.4323, 41.2642, 25.5707],
    panels: { trunk: 1 }, hide: /^MR2_Trunk_ServiceCarpet(?:_|$)/,
  },
  "E51 transaxle case and cooler pipe": {
    theta: 0.4475, phi: 2.76826, roll: -0.49304,
    distance: 48.1138, fov: 24.8931, center: [-59.6588, 18.6747, -7.5488],
    hideFloor: true, hide: /^MR2_(?:Body_Shell_Unibody|SwayBar)_/,
  },
  "Transaxle output and rear CV axle": {
    theta: 0.89, phi: 2.12, roll: -0.06,
    distance: 33.0, fov: 48.0, center: [-68.9, 17.5, -20.1],
    hideFloor: true, hide: /^MR2_(?:Body_Shell_Unibody|SwayBar)_/,
  },
  "Rear drivetrain viewed from below": {
    theta: 3.13889, phi: 2.48198, roll: 0.00732,
    distance: 129.0271, fov: 27.3764, center: [-77.3422, 17.106, 2.9297],
    hideFloor: true, hide: /^MR2_Body_Shell_Unibody_/,
  },
};
