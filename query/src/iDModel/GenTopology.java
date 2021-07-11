package iDModel;

import indoor_entitity.D2Ddistance;
import indoor_entitity.Graph;
import indoor_entitity.IndoorSpace;

import java.io.IOException;
import java.util.HashMap;

public class GenTopology {
    /**
     * generate connectivityTier
     */
    public void genTopology() throws IOException {
        int partitionSize = IndoorSpace.iPartitions.size();
        for (int i = 0; i < partitionSize; i++) {

            // D2D distance Matrix
            HashMap<String, D2Ddistance> hashMap = IndoorSpace.iPartitions.get(i).getD2dHashMap();
            IndoorSpace.iD2D.putAll(hashMap);
//			System.out.println("partition " + IndoorSpace.iPartitions.get(i).getmID() + " has size " + IndoorSpace.iPartitions.get(i).getD2dHashMap().size());

            // set weight center of all of the partitions
            IndoorSpace.iPartitions.get(i).setCenter();

            // partition's distance matrix
            DistMatrix distMatrix = new DistMatrix(IndoorSpace.iPartitions.get(i).getmID(), true);
            IndoorSpace.iPartitions.get(i).setDistMatrix(distMatrix);

            // partition's connectivity tier
            Topology topology = new Topology(IndoorSpace.iPartitions.get(i).getmID());
            IndoorSpace.iPartitions.get(i).setTopology(topology);

            // add all the partitions into the Graph
            Graph.Partitions.put(i, IndoorSpace.iPartitions.get(i));
        }


    }
}
