using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveDummyClass : MonoBehaviour
{
    // Start is called before the first frame update
    public List<GameObject> dummies;
    public float[] speed;
    public GameObject walkway;

    public List<Animator> animators;
    //Vector3 translateVec = Vector3.zero;
    void Start()
    {
        Debug.Log(dummies.Count);
        animators = new List<Animator>();
        speed = new float[dummies.Count];
        for (int i = 0; i < dummies.Count; i++)
        {
            animators.Add(dummies[i].GetComponent<Animator>());
            float speedCurr = Random.Range(0.8f, 1.5f);
            speed[i] = speedCurr;
        }
        
    }

    // Update is called once per frame
    void Update()
    {
        for (int i = 0;i < animators.Count; i++)
        {
            animators[i].speed = speed[i];


            dummies[i].transform.Translate(dummies[i].transform.forward * speed[i] * Time.deltaTime,Space.World);
        }
        
        //animator.SetFloat("Speed", speed * Time.deltaTime);
        
    }
}
