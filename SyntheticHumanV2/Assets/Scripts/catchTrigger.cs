using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class catchTrigger : MonoBehaviour
{
    GameObject manager;
    void Start()
    {
        manager = GameObject.Find("Manager");
    }

    private void OnTriggerEnter(Collider other)
    {
        //Debug.Log("TEST");
        
        CheckReach currScript = other.gameObject.transform.GetComponent<CheckReach>();

        currScript.hasReached = true;

        manager.GetComponent<InitializeMannequin>().allMannReached[currScript.nameObj] = true;

        manager.GetComponent<InitializeMannequin>().numReached += 1;

        Debug.Log($"Mann {currScript.nameObj} reached");
        
    }
}
