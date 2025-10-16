using System;

class Program
{
    static void Main()
    {
        string[] names = { "Rikke", "Jens", "Stefan", "Anne" };

        for (int i = 0; i < names.Length; i++)
        {
            Console.WriteLine(names[i]);
        }



        foreach (string name in names)
        {
            Console.WriteLine(name);
        }
        
    }
    
}