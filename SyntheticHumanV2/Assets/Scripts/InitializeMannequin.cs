using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

public class InitializeMannequin : MonoBehaviour
{
    public GameObject mannequinePref;
    public GameObject endPointPref;

    public List<GameObject> allMannequines;
    public int numMannequines;
    public int maxNumMannequines;

    public GameObject centerPoint;
    public GameObject leftSideObj;
    public GameObject rightSideObj;

    public float maxOffsetAngle = 30f;

    public List <GameObject> allEnds;

    public bool reset = false;

    public Dictionary<string, bool> allMannReached = new Dictionary<string, bool>();

    public int numReached = 0;

    public bool isRandNum = false;

    public bool startWithPlay = false;

    public int numCaptures = 30;
    public int currCapture = 1;

    public GameObject separatorCube;

    public float separatorFramesMax = 3f;
    

    // Start is called before the first frame update
    void Start()
    {

        numMannequines = randomizeNumMannequines(numMannequines, isRandNum, maxNumMannequines);
        createMannequines(numMannequines);
        
    }

    List<Vector3> selectRandomPos()
    {
        int leftOrRight = Random.Range(0, 2);
        Vector3 currPos = Vector3.zero;
        Vector3 currEndPos = Vector3.zero;
        if (leftOrRight == 0)
        {
            currPos.x = leftSideObj.transform.position.x;

            currEndPos.x = rightSideObj.transform.position.x;
        }
        else {
            currPos.x = rightSideObj.transform.position.x;

            currEndPos.x = leftSideObj.transform.position.x;
        }

        currPos.z = Random.Range(-4f, 4f);
        currPos.y = -1f;

        currEndPos.z = Random.Range(-4f, 4f);
        currEndPos.y = -1f;

        List<Vector3> startEnd = new List<Vector3>();
        startEnd.Add(currPos);
        startEnd.Add(currEndPos);
        return startEnd;
    } 

    Quaternion selectRandomRot(Quaternion currRot) 
    {

        // Convert the random offset to Euler angles
        Vector3 randomEulerAngles = currRot.eulerAngles;

        // Apply the maximum offset angle to one of the Euler angles
        randomEulerAngles.y += Random.Range(-maxOffsetAngle, maxOffsetAngle);

        // Convert the Euler angles back to a quaternion
        Quaternion finalRotation = Quaternion.Euler(randomEulerAngles);

        // Blend the desired rotation with the random offset
        //finalRotation = Quaternion.Lerp(currRot, finalRotation, Random.value);

        
        return finalRotation;


    }

    int randomizeNumMannequines(int numMan, bool isRand = false, int maxRand = 5)
    {
        if (isRand)
        {
            numMan = Random.Range(1, maxRand);

        }

        return numMan;
    }

    void createMannequines(int numMan)
    {

        

        allMannequines = new List<GameObject>();

        allEnds = new List<GameObject>();

        for (int i = 0; i < numMan; i++)
        {
            List<Vector3> startEnd = selectRandomPos();

            Vector3 currPos = startEnd[0];
            Vector3 endPos = startEnd[1];

            Quaternion currRot = Quaternion.LookRotation(endPos - currPos);

            //currRot = selectRandomRot(currRot);

            Debug.Log(centerPoint.transform.position - currPos);

            GameObject currMann = Instantiate(mannequinePref, currPos, currRot);

            GameObject currEnd = Instantiate(endPointPref, endPos, Quaternion.identity);

            allMannequines.Add(currMann);
            allEnds.Add(currEnd);

            //allMannReached.Add($"mann_{i}", false);
            currMann.GetComponentInChildren<CheckReach>().nameObj = $"mann_{i}";

        }
    }


    void startAnimations (List<GameObject> allMannequines)
    {
        for (int i = 0; i < allMannequines.Count; i++)
        {
            Animator anim = allMannequines[i].GetComponent<Animator>();

            anim.SetInteger("MotionType", 0);

            anim.SetInteger("WalkInd", Random.Range(0, 12));
            anim.speed = Random.Range(0.6f, 1.4f);

        }
    }

    void destroyPeople(List<GameObject> allMannequines, List<GameObject> allEnds)
    {
        for (int i = 0; i < allMannequines.Count; i++)
        {
            Destroy(allMannequines[i]);
            Destroy(allEnds[i]);
        }

        allMannequines = new List<GameObject>();

        allEnds = new List<GameObject>();
    }

    IEnumerator RemoveAfterSeconds(float seconds, GameObject obj)
    {
        yield return new WaitForSeconds(seconds);
        obj.SetActive(false);
    }


    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space) || startWithPlay)
        {
            //for (int i = 0; i < allMannequines.Count; i++)
            //{
            //    Animator anim = allMannequines[i].GetComponent<Animator>();

            //    anim.SetInteger("MotionType", 0);

            //    anim.SetInteger("WalkInd", Random.Range(0,12));
            //    anim.speed = Random.Range(0.6f, 1.4f);

            //}
            startAnimations(allMannequines);
            startWithPlay = false;

            separatorCube.SetActive(true);
            StartCoroutine(RemoveAfterSeconds(separatorFramesMax, separatorCube));
        }

        //for (int i = 0; i < allMannequines.Count; i++)
        //{
        //    GameObject endPoint = allEnds[i];
        //    Vector3 dir = endPoint.transform.position- allMannequines[i].transform.position;
        //    Debug.DrawRay(allMannequines[i].transform.position, dir*1000f, Color.green);

        //}



        if (numReached >= numMannequines)
        {
            reset = true;
            numReached = 0;
            Debug.Log("All Reached");

            currCapture++;
        }


        if (reset && (currCapture <= numCaptures))
        {
            destroyPeople(allMannequines, allEnds);
            numMannequines = randomizeNumMannequines(numMannequines, isRandNum,maxNumMannequines);
            createMannequines(numMannequines);
            startAnimations(allMannequines);

            separatorCube.SetActive(true);
            StartCoroutine(RemoveAfterSeconds(separatorFramesMax, separatorCube));

            reset = false;
        }

        if (currCapture > numCaptures)
        {
            EditorApplication.isPlaying = false;
        }

    }
}
