# Dynamic progress monitoring - project plan 
EDGAR is an equation discovery engine that uses an evolutionary algorithm involving LLMs to come up with equations that best describe a given dataset. 

Currently, evolutionary history is generated at the end of each run by creating a family tree. 
The family tree is generated as an HTML file based on a JSON file that is created and updated continuously. 

We now want to create another monitoring tool which allows us to track the progress made by EDGAR dynamically. 
In the monitoring tool, we want to see 
1. Number of iterations processed so far 
2. Progress of training loss so far (no test loss available until the end of the process)
3. Effect of gradient descent on model training loss vs model training loss with just parameter estimator 

I believe all of this information is already involved in the JSON file, so we just need to create a function like : 
    def create_dynamic_progress_update(JSON_FILE, OUTPUT_DIR): 
        if JSON_FILE is empty : 
            return None
        else : 
            # 1. Parse JSON file to extract all available information. Find out how far we are in the progress (n_iter / total_iter)
            # - extract the following detail of each program (this is very similar to family tree)
            # -- n_iter, island, batch, training loss, complexity penalty, exploration_expoitation_mode, llm name, learning rate, model code string, parameter estimator, parent program identity, prompt string used to generate this model, image prompt used to generate the program, model visualisation  
            # 2. Save 2 different HTML files - one showing the prgress of training loss, another showing the effect of gradient descent 

            # save files 
        return None 

## Step 1
0. Note that every program is evaluated by their score, which is a summation of their loss and the complexity penalty. 
Make sure that they are stored separately in the JSON file. If the complexity penalty is not already stored, add it.  
1. Rename src/family_tree to src/progress_report and let family tree building be one of the possible functions available 
2. Find out what information is available dynamically (i.e. in the middle of a run) in the JSON file and how it can be retrieved. How is information appended to the JSON file? Is it safe to read it at any point? Do we expect it to be syntactically correct? 
3. The existing function in family_tree where it reads the JSON file should be helpful, so modularise that without changing any behaviour for family_tree. 
4. Write create_dynamic_progress_update. Follow "Specific guideline for 2" below for specific instructions. 

## Specific guideline for 2 - Progress of training loss of all programs created across all islands. 
- Create a plot where the x axis is the discrete n_iter and y axis is the negative loss. 
- Mark each program generated as a scatter plot. For visibility create small offsets for each island idx. Colour-code programs by their island. 
- Make it possible to click any of the nodes to reveal their details - much like the way we've done for the family tree. 
- Extract information about their loss and the complexity penalty. Create a toggle to determine whether to add the complexity penalty or not. 
- Create a toggle to view only specific islands. Best if we can tick islands to view so we can select any arbitrary combination of islands at a given time. 

## Step 2 
1. When you hover over any particular node, highlight their lineage. 
2. An extra plot with the x axis as the number of parameters and the y axis as the loss, with a series of diagonal line that allows us to visualise losses that lead to the same score (because score = loss + penalty and penalty = lambda * n_params), so you can draw y = lambda x on this plot to see which models with differing loss have similar scores due to this penalty. 