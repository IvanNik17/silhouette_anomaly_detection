using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckReach : MonoBehaviour
{

    public bool hasReached = false;
    public string nameObj = string.Empty;
    //GameObject manager;
    //// Start is called before the first frame update
    //void Start()
    //{
    //    manager = GameObject.Find("Manager");
    //}

    //// Update is called once per frame
    //void Update()
    //{
        
    //}


    //private void OnTriggerEnter(Collider other)
    //{
    //    Debug.Log("TEST");
    //    if (other.gameObject.CompareTag("endPoint"))
    //    {
    //        hasReached = true;

    //        manager.GetComponent<InitializeMannequin>().allMannReached[nameObj] = true;

    //        manager.GetComponent<InitializeMannequin>().numReached += 1;

    //        Debug.Log($"Mann {nameObj} reached");
    //    }
    //}

}
