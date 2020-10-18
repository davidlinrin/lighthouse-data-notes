"""
 - Connect to the hr.db (stored in supporting-files directory) with sqlite3 
 - Write a query to find the names (first_name, last_name) of the employees who have a manager who works for a department based in the United States. 
 

Expected columns:
    - first_name	
    - last_name	

Notes:
    - Use tables employees, departments and locations
    - You shouldn’t use JOINs here. 
    - You can connect to DB from Jupyter Lab/Notebook, explore the table and try different queries
    - In the variable 'SQL' store only the final query ready for validation 
"""


#SQL = 'select first_name, last_name from employees  where department_id in (select department_id from departments where location_id in (select location_id from locations where country_id = "US"))'



SQL = 'select first_name, last_name, manager_id from employees where manager_id IN (select manager_id from employees where department_id IN (SELECT department_id FROM departments WHERE location_id IN (select location_id FROM locations WHERE country_id = "US")))'
