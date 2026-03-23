for j in $(squeue -h --qos=job_gpu_caim -t RUNNING -o %i); do

  scontrol show job $j | awk -v jid="$j" '

    /UserId=/{u=$1; sub("UserId=","",u); sub("\\(.*","",u)}

    /Partition=/{p=$1; sub("Partition=","",p)}

    /NodeList=/{n=$1; sub("NodeList=","",n)}

    /RunTime=/{rt=$1; sub("RunTime=","",rt)}

    /ReqTRES=/{r=$0; sub(".*ReqTRES=","",r)}

    END{print jid, u, p, n, rt, r}

  '

done
