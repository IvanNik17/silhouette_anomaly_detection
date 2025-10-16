using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConcentricHemispheres : MonoBehaviour
{
    public GameObject cameraPrefab;
    public Transform centerTarget;
    public int numHemispheres = 3;
    public int camerasPerHemisphere = 30;
    public float minRadius = 5f;
    public float maxRadius = 15f;
    [Range(0f, 180f)] public float arcAngle = 90f;

    void Start()
    {
        GenerateConcentricHemispheres();
    }

    void GenerateConcentricHemispheres()
    {
        for (int h = 0; h < numHemispheres; h++)
        {
            float t = (numHemispheres > 1) ? (float)h / (numHemispheres - 1) : 0f;
            float radius = Mathf.Lerp(minRadius, maxRadius, t);

            // Calculate number of rows based on sqrt heuristic
            int numRows = Mathf.CeilToInt(Mathf.Sqrt(camerasPerHemisphere));
            int placedCameras = 0;

            for (int row = 0; row < numRows; row++)
            {
                // Evenly distribute phi within arc
                float phi = Mathf.Deg2Rad * arcAngle * ((float)row / (numRows - 1));

                // How many cameras in this row?
                int numCols = Mathf.CeilToInt((float)camerasPerHemisphere / numRows);

                for (int col = 0; col < numCols && placedCameras < camerasPerHemisphere; col++)
                {
                    float theta = (2f * Mathf.PI * col) / numCols;

                    // Convert spherical to Cartesian
                    float x = radius * Mathf.Sin(phi) * Mathf.Cos(theta);
                    float y = radius * Mathf.Cos(phi);
                    float z = radius * Mathf.Sin(phi) * Mathf.Sin(theta);

                    Vector3 position = centerTarget.position + new Vector3(x, y, z);

                    GameObject camObj = Instantiate(cameraPrefab, position, Quaternion.identity, transform);
                    camObj.transform.LookAt(centerTarget.position);

                    placedCameras++;
                }
            }
        }
    }
}
