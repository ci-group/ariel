export default {
    template: `
    <button @click="calculate" :style="{ background: value > 0 ? '#bf8' : '#eee', padding: '8px 16px', borderRadius: '4px' }">
    <strong>{{title}}: {{total}}</strong>
    </button>
`,
    props: {
        title: String,
    },
    data() {
        return {
            total: 0,
            graph: null,
            sumNode: null,
        };
    },
    mounted() {
        this.graph = new LGraph();
        let canvas = new LGraphCanvas("#node-editor", this.graph);

        // register node type
        function sum(x, y) {
            var total = x + y;
            return total;
        }
        LiteGraph.wrapFunctionAsNode("ec/sum", sum, ["Number", "Number"], "Number");

        // add node
        this.sumNode = LiteGraph.createNode("ec/sum");
        this.sumNode.pos = [400, 200];
        this.graph.add(this.sumNode);

        // add input nodes
        let node1 = LiteGraph.createNode("basic/const", 1);
        node1.pos = [100, 200];
        this.graph.add(node1);

        let node2 = LiteGraph.createNode("basic/const", 2);
        node2.pos = [100, 300];
        this.graph.add(node2);

        // connect nodes
        node1.connect(0, this.sumNode, 0);
        node2.connect(0, this.sumNode, 1);

        // start the graph
        this.graph.start();

        // Update count
        this.graph.runStep(1);
        this.total = this.sumNode.getOutputData(0);
    },
    methods: {
        calculate() {
            this.graph.runStep(1);
            this.total = this.sumNode.getOutputData(0);
            console.log("Total:", this.total);
            this.$emit("calculate", this.total);
        }
    },
};